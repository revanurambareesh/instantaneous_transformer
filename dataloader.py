from os import execle
from re import sub
from typing_extensions import runtime
from numpy import split
from numpy.linalg.linalg import det
from imresize import imresize
from globals import *
logger = Logging().get(__name__, args.loglevel)

from torch.utils.data import DataLoader
from utils import get_all_frames, detrend_filter
from skimage import io
import scipy.io as sio
import scipy.signal
import scipy.ndimage

torch.multiprocessing.set_sharing_strategy('file_system')


def clean_bvp_signal(path_file, frameidx, seqlen, subject):
    phys_1k = np.loadtxt(path_file, delimiter='\n')
    phys = np.array([np.mean(phys_1k[i:i+40]) for i in range(0, len(phys_1k)-40, 40)])
    phys = phys[frameidx: frameidx + seqlen]
    assert len(phys) > 0, ['Missing Phys', subject]
    phys = detrend_filter(phys, cumsum=False)
    y_phys = norm_sig(phys)
    return y_phys


def post_process_mean_std_vid(seqlen, seqX):
    seqOri = np.copy(seqX)
    for i in range(seqlen-1):
        diff = (seqX[i+1] - seqX[i]) / (seqX[i+1] + seqX[i] + 3e-5)
        seqX[i] = diff
    
    seqX[-1] = seqX[-2]
    seqX = seqX / (np.std(seqX) + 3e-5 )
    assert seqX.shape == seqOri.shape

    seqOri = seqOri - np.mean(seqOri)
    seqOri = seqOri / np.std(seqOri)

    seqX = np.concatenate([seqX, seqOri], -1)

    return seqX


def post_process_signal_first_derivative(seqlen, sig):
    sigdiff = np.copy(sig)
    for i in range(seqlen-1):
        sigdiff[i] = (sig[i+1] - sig[i])
    
    sigdiff[-1] = sigdiff[-2]
    sigdiff = sigdiff / max(sigdiff)
    
    return sigdiff


def get_appearance_motion(image_batch):
    assert image_batch.shape[1:] == torch.Size([3, 36, 36]) and len(image_batch.shape) == 4

    lshifted = torch.cat([image_batch, torch.zeros(1,3,36,36).cuda()], 0)[1:]
    motion = (image_batch/2 - lshifted/2) / (image_batch/2 + lshifted/2 + 1e-5)
    motion[motion > 3] = 3
    motion[motion < -3] = -3
    motion = motion[:-1]
    motion = (motion - motion.mean()) / motion.std()

    appearance = image_batch[:-1]
    estimated_mean, estimated_std = torch.mean(appearance), torch.std(appearance)
    appearance = (appearance - estimated_mean) / estimated_std

    return appearance, motion



class V4V_Dataset(DataLoader):
    def __init__(self, split, use_cache=False):
        assert split in ['training', 'validation', 'testing'], ['split issue', 'training']
        self.seqlen = args.seqlen
        self.datapath = osj(data_dir, 'v4v_dataset')
        self.use_cache = use_cache
        self.split = split

        self.gt_validation = {}
        self.gt_test = {}

        logger.info(f'V4V Dataset with split: {self.split}')

        self.data_seq_list = self.__config()

    def __config(self):
        if self.split == 'training':
            subjects = glob(osj(self.datapath, 'Training', 'Videos', 'Training', 'vids', '*.mkv'))
        elif self.split == 'validation':
            subjects = glob(osj(self.datapath, 'Validation', '*.mkv'))
        elif self.split == 'testing':
            subjects = glob(osj(self.datapath, 'Test', '*.mkv'))
        else:
            raise NotImplementedError()

        lens, seqs = len(subjects), []


        subject_list = sorted(subjects)
        np.random.seed(123)
        if self.split == 'training':
            np.random.shuffle(subject_list)

        if self.split == 'training':
            subject_list = subject_list
        elif self.split == 'validation':
            subject_list = subject_list
        elif self.split == 'testing':
            subject_list = subject_list


        for vidname in tqdm(subject_list):
            subjname = vidname.split('/')[-1]
            ### ### ###
            cap = cv2.VideoCapture(vidname); length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ### ### ###

            if length < self.seqlen:
                logger.debug(f'Subject {vidname} has < {self.seqlen} frames with {length} frames')
                continue
                
            for i in range(length // self.seqlen - 1):
                seqs.append((vidname, i * self.seqlen))
                logger.debug(f'{vidname}: {i*self.seqlen}')

        if self.split == 'validation' or self.split == 'testing':
            fname = 'gt_validation.txt' if self.split == 'validation' else 'gt_test.txt'
            gtfile = osj(self.datapath, 'private', fname)
            with open(gtfile) as f:
                start_idx = 0 if PHYS_TYPE == 'HR' else 1
                for line in f.readlines()[start_idx::2]:
                    line = line.split(',')
                    subj, phys, hr = line[0].strip(), line[1].strip(), np.array(line[2:], dtype=np.float32)
                    assert phys == PHYS_TYPE
                    if self.split == 'validation':
                        self.gt_validation[subj] = hr.astype(np.float32)
                    elif self.split == 'testing':
                        self.gt_test[subj] = hr.astype(np.float32)

        return seqs



    def postprocess(self, datadict, subject, frameidx):
        debug_dict = {'subject': subject, 'frameidx': frameidx}

        ####

        debug_dict['oriX'] = np.copy(datadict['X'])
        datadict['X'] = post_process_mean_std_vid(self.seqlen, datadict['X'])
        datadict['y_bp'] = post_process_signal_first_derivative(self.seqlen, datadict['y_bp'])

        assert len(datadict['y_bp']) == len(datadict['X'])
        return datadict, debug_dict

    def __getitem__(self, idx):
        vidpath, frameidx = self.data_seq_list[idx]
        
        subject = vidpath.split('/')[-1].split('.')[0]

        #### 2. Video frames
        subjname = vidpath.split('/')[-1]
        subj = subjname.split('.')[0]
        
        frames = get_all_frames(vidpath, frameidx, frameidx+self.seqlen, (36, 36))
        assert len(frames) == self.seqlen, f'Turns out {len(frames)}=={self.seqlen} for {subj} with {frameidx}'
        dataX = np.stack(frames)

        if PHYS_TYPE == 'HR':
            #### 3. BVP signal with butterworth
            subjbpname = subject.replace('_', '-')
            if self.split == 'training':
                subjpath = osj(self.datapath, 'Training', 'BP_raw_1KHz', f'{subjbpname}-BP.txt')
            else:
                subjpath = osj(self.datapath, 'private', 'txt_bp', f'{subjbpname}-BP.txt')
            
            if self.split in ['training', 'validation']:
                y_phys = clean_bvp_signal(subjpath, frameidx, self.seqlen, subject)
            else:
                y_phys = np.zeros(dataX.shape[0])

            #### 4. Heart rate
            if self.split == 'training':
                hr = np.loadtxt(osj(self.datapath, 'Training', 'Physiology', f'{subject}.txt'), delimiter=',', dtype=str)
                hr = hr[0, 2:].astype(np.float32)
                hr = hr[frameidx:frameidx+self.seqlen]
            elif self.split == 'validation':
                hr = self.gt_validation[f'{subject}.mkv'][frameidx:frameidx+self.seqlen]
            elif self.split == 'testing':
                hr = self.gt_test[f'{subject}.mkv'][frameidx:frameidx+self.seqlen]
            else:
                raise NotImplementedError()
        else:
             #### 3. RR signal with butterworth
            subj_head, subj_tail = subject.split('_')
            if self.split in ['training', 'validation']:
                subjpath = osj(data_dir, 'bp4dPhys', subj_head, subj_tail, 'Respiration Rate_BPM.txt')
                y_phys = clean_bvp_signal(subjpath, frameidx, self.seqlen, subject)
            else:
                y_phys = np.zeros(dataX.shape[0])

            #### 4. Respiration rate -- Variable name has been reused.
            if self.split == 'training':
                hr = np.loadtxt(osj(self.datapath, 'Training', 'Physiology', f'{subject}.txt'), delimiter=',', dtype=str)
                hr = hr[1, 2:].astype(np.float32)
                hr = hr[frameidx:frameidx+self.seqlen]
            elif self.split == 'validation':
                hr = self.gt_validation[f'{subject}.mkv'][frameidx:frameidx+self.seqlen]
            elif self.split == 'testing':
                hr = self.gt_test[f'{subject}.mkv'][frameidx:frameidx+self.seqlen]
            else:
                raise NotImplementedError()

        #### 5. Length
        assert len(dataX) == len(y_phys) and len(dataX) == len(hr), \
                ['Length validation', f'{subject} :: {len(dataX)}--{len(y_phys)}--{len(hr)}']

        retdict = {'X': dataX, 'y_bp': y_phys, 'y_hr': hr}
        return self.postprocess(retdict, subject, frameidx)


    def __len__(self):
        return len(self.data_seq_list)


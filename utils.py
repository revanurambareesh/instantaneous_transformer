from globals import *
logger = Logging().get(__name__)

import shutil
import scipy.signal
from imresize import imresize

def get_all_frames(video_path, start=0, end=None, resize=None):
  frames = []

  logger.debug(f'Reading the video from {video_path}')
  cap = cv2.VideoCapture(video_path)
  frame_count = 0

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
      # cv2.imshow('Frame',frame)
      if frame_count >= start:
        if end == None or frame_count < end:
        #   ### This thing below is needed for FRONTALIZED VIDS only
        #   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          if resize != None:
            # img.resize((width, height), Image.ANTIALIAS)
            frame = imresize(frame, output_shape=resize)

          if np.sum(frame) == 0:
              logger.warn(f'Removing the black frames')  
          else:
              frames.append(frame)
      elif frame_count > end:
          break

      frame_count += 1

      # Press Q on keyboard to  exit
      # if cv2.waitKey(25) & 0xFF == ord('q'):
      #   break
    # Break the loop
    else:
      break
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv2.destroyAllWindows()

  logger.debug('Video closed. Returning frames.')

  return frames


def resize_all_frames(frames, scale_factor):
    assert scale_factor > 0

    resized_frames = []
    for frm in frames:
        size = (scale_factor * frm.shape[0], scale_factor * frm.shape[1])
        resized_frames.append(cv2.resize(frm, size))

    return resized_frames


def create_dir(directory):
    if not os.path.exists(directory):
        logger.info(f'Creating directory {directory}')
        os.makedirs(directory)


import tensorboardX
from tensorboardX import SummaryWriter
import os

class SummaryLogger:
    def __init__(self, summaries_path):
        self.train_writer = SummaryWriter(summaries_path, flush_secs=2)

    def log_errors(self, summary_dict, xaxis_value=None, phase='train'):
            for x in list(sorted(summary_dict.keys())):
                try:
                    val = det_cpu_npy(summary_dict[x])
                except:
                    val = x
                    
                if val == np.nan:
                    logger.error('Nan encountered')
                self.train_writer.add_scalar(x, summary_dict[x], xaxis_value)

    def add_embedding(self, *args, **kwargs):
        self.train_writer.add_embedding(*args, **kwargs)


###########

def adjust_learning_rate(optimizer, epoch):
    ## Optional code block added here for reference. 

    # lr = args.lr
    # # if epoch >= 20 and epoch < 40:
    # #     lr = args.lr * 0.1
    # # if epoch >=120:
    # #     lr = args.lr * 0.01


    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # pass
    pass



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = osj(weights_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = osj(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osj(directory, 'model_best.pth.tar'))



def detrend_filter(pred, cumsum=True):
    if cumsum:
        yptest = np.cumsum(pred).squeeze()
    else:
        yptest = pred

    lam = 10
    fs = args.vidfps

    b, a = scipy.signal.butter(2, [LOW_HR_FREQ/fs*2, HIGH_HR_FREQ/fs*2], 'bandpass')

    T = len(yptest)
    I = scipy.sparse.eye(T)
    assert T>2
    D2 = scipy.sparse.spdiags((np.ones((T-2, 1))*[1, -2, 1]).T,range(0,3),T-2,T).toarray()
    D2[-1, -1] = 1
    D2[-1, -2] = -2
    D2[-2, -2] = 1
    D2 = scipy.sparse.csr_matrix(D2)

    sr = scipy.sparse.csr_matrix(yptest)
    divi = lambda a, b: np.dot(a.toarray() ,np.linalg.pinv(b.toarray()))
    temp = I * sr.T - divi(sr, (I + (lam**2)* D2.T * D2).T).T

    nZ = (temp - np.mean(temp))/np.std(temp)
    yptest_sub2 = scipy.signal.filtfilt(b,a,nZ.T,padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    return yptest_sub2










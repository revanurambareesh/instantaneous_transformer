from globals import *
logger = Logging().get(__name__, args.loglevel)

from network import *


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.netcomp = {}

        self.tnet = Tnet()
        self.class_token = None
        self.fps = args.vidfps
        self.batch_size = args.batch_size
        self.criterion = NegativeMaxCrossCorr(args)

    
    def bp_loss(self, x_frames, bp_signal, hr_signal, optimizer, scheduler):
        assert x_frames.shape[:2] == hr_signal.shape
        assert bp_signal.shape == hr_signal.shape

        self.feats = {}
        out, tnet_debug_dict = self.tnet(x_frames, bp_signal)
        self.feats['tnet'] = out

        if optimizer is not None:
            optimizer.zero_grad()  
       
        ### MCCLoss
        loss = self.criterion(out, bp_signal)

        if optimizer is not None and torch.isnan(out).sum() == 0:
            try:
                loss.backward()
            except:
                bp()
                
            optimizer.step() 

        if scheduler is not None:
            scheduler.step()
        
        return loss, tnet_debug_dict


    def forward_eval(self, x):
        assert not self.training
        raise NotImplementedError()

    
    def forward(self, x, bp_signal):
        pass



'''
Credits: https://github.com/ToyotaResearchInstitute/RemotePPG
'''
tr = torch

class NegativeMaxCrossCov(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels):
        # Normalize
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        # preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        # labels_fft = torch.fft.rfft(labels_pad, dim=-1)
        N = 4*preds_pad.shape[-1] if PHYS_TYPE == 'HR' else 8*preds_pad.shape[-1]
        preds_fft = torch.fft.rfft(preds_pad, dim=-1, n = N)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1, n = N)
        freqs = torch.fft.rfftfreq(n=N) * self.Fs

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        # freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:,use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:,zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        # Zero out irrelevant freqs
        X[:,zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio
        

        return -max_cc
    
class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, args):
        super(NegativeMaxCrossCorr, self).__init__()
        Fs = args.vidfps
        high_pass = HIGH_HR_FREQ * 60
        low_pass = LOW_HR_FREQ * 60
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        # return output
        return output.mean()










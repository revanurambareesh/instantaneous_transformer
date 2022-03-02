import cv2
import os, argparse, sys
from os.path import join as osj
from glob import glob
from os import listdir as ls
import numpy as np
import scipy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, models
from pdb import set_trace as bp
from natsort import natsorted as nsort

import scipy.io as sio
import skimage.io as skio

import logging, coloredlogs
from deprecated import deprecated

root_dir = '../..'

# Training settings
parser = argparse.ArgumentParser(description='Instantaneous_Transformer')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--epochs', type=int, default=1000, metavar='N')
parser.add_argument('--enough_epochs', type=int, default=70, metavar='N')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N')
parser.add_argument('--lr', type=float, default=9e-4, metavar='LR')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=250, metavar='N')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--name', default='exp_001', type=str)
parser.add_argument('--datadir', default=f'{root_dir}/data', type=str)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--loglevel', default='INFO', type=str)
parser.add_argument('--seqlen', type=int, default=100, metavar='N')
parser.add_argument('--gpu', type=int, default=0, metavar='N')
parser.add_argument('--vidfps', type=int, default=25, metavar='N')
parser.add_argument('--numlayer', type=int, default=2, metavar='N')
parser.add_argument('--valfreq', type=int, default=5, metavar='N')
parser.add_argument('--phys', type=str, default='HR')


args, _ = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not args.no_cuda:
    torch.cuda.set_device(f'cuda:{args.gpu}')

exp_name = args.name
data_dir = args.datadir
summaries_dir = osj(root_dir, 'summaries', exp_name)
weights_dir = osj(root_dir, 'weights', exp_name)

if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)


PHYS_TYPE = args.phys
assert PHYS_TYPE in ['HR', 'RR']

LOW_HR_FREQ = 0.7 if PHYS_TYPE == 'HR' else 8 / 60
HIGH_HR_FREQ = 2.5 if PHYS_TYPE == 'HR' else 22 / 60

print(f'THE PHYS_TYPE IS {PHYS_TYPE}')

det_cpu_npy = lambda x : x.detach().cpu().numpy()
norm_sig = lambda x: np.reshape((x-x.mean()) / np.max(np.abs(x-x.mean())), -1)

def tqdm(iterator):
    from tqdm import tqdm
    return tqdm(iterator, bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')


class DuplicateFilter(logging.Filter):

    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


class Logging:
    __logger_dict = {}

    def get(self, name, level='INFO'):
        assert level == 'INFO', 'works with only INFO'
        if name not in self.__logger_dict:
            logger = logging.getLogger(name)
            logger.addFilter(DuplicateFilter())
            logger.setLevel(logging.DEBUG)
            h = logging.FileHandler("logs/debug.log")
            formatter = logging.Formatter(
                    f'%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s [%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                )
            h.setFormatter(formatter)
            h.setLevel(logging.DEBUG)
            logger.addHandler(h)
            logger.propagate = False


            console_logging = logging.StreamHandler(sys.stdout)
            console_logging.setLevel(logging.INFO)
            console_logging.setFormatter(formatter)
            logger.addHandler(console_logging)

            self.__logger_dict[name] = logger
            coloredlogs.install(level=level, logger=logger)
    
        return self.__logger_dict[name]



def set_highlighted_excepthook():
    import sys, traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
    formatter = TerminalFormatter()

    def myexcepthook(type, value, tb):
        tbtext = ''.join(traceback.format_exception(type, value, tb))
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook


set_highlighted_excepthook()


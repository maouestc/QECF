import argparse
import os
from dataset import *
from solver import Solver
import torchvision.transforms as transforms
from model_R1 import *
import torch.nn as nn
import random
import h5py

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = FrameQE()
    checkpoint = torch.load('stdf_scc_37.pt')
    model.load_state_dict(checkpoint.state_dict())

    solver = Solver(model, check_point='', batch_size=1)
    solver.test_r1(train_dir='/home/shelei/hjw/MFQE_hjw/testdata/QP37/FatekaleidQ37_1920_1080_2.yuv',
                   label_dir='/home/shelei/hjw/MFQE_hjw/testdata/raw/Fatekaleid_1920_1080_2.yuv',
                   h=1080, w=1920, tot_frm=300)

if __name__ == '__main__':
    main()


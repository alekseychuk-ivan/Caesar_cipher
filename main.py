import os
import random
import torch
import numpy
import time
from torch.utils.data import Dataset
import argparse
from utils.module import *


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # choice device


# parse our input parameters
parse = argparse.ArgumentParser()
parse.add_argument('--file', '-f', help='Path to file and file name', default="./input/onegin.txt")
parse.add_argument('--epoch', '-e', help='Numbers of epoch', default=20, type=int)
parse.add_argument('--val', '-v', help='Size of validation sample', default=0.1, type=float)
parse.add_argument('--test', '-t', help='Size of test sample', default=0.2, type=float)
parse.add_argument('--hidden', '-hi', help='Number of hidden layer', default=100, type=int)
parse.add_argument('--emb', '-emb', help='Embedded dimensional', default=150, type=int)
parse.add_argument('--outsize', '-o', help='Number of output size', default=150, type=int)
parse.add_argument('--caesar', '-c', help='Caesar offset', default=1, type=int)

args = parse.parse_args()


args = parse.parse_args()
FILE_NAME = args.file
NUM_EPOCHS = args.epoch
test_size = args.test
val_size = args.val
hidden_size = args.hidden
out_size = args.outsize
embedd_dim = args.emb
CAESAR_OFFSET = args.caesar

# start training model and verify
train(NUM_EPOCHS=NUM_EPOCHS, test_size=test_size, val_size=val_size, FILE_NAME=FILE_NAME, hidden_size=hidden_size,
      out_size=out_size, embedd_dim=embedd_dim, CAESAR_OFFSET=CAESAR_OFFSET, DEVICE=DEVICE)

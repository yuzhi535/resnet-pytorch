import torch
import torchmetrics
from argparse import ArgumentParser

def train_fn():
    pass

def val_fn():
    pass

def train():
    pass

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--batch-size', '-bs', type=int, default=8, help='batch size')
    argparser.add_argument('--num-workers', '-nw', type=int, default=4, help='number of workers')
    argparser.add_argument('--pin-memory', '-pm', action='store_true', help='pin memory')
    argparser.add_argument('--augment', '-aug', action='store_true', help='augment')
    argparser.add_argument('--random-seed', '-rs', type=int, default=42, help='random seed')
    argparser.add_argument('--valid-rate', '-vr', type=float, default=0.1, help='valid rate')
    
    
    
    train()
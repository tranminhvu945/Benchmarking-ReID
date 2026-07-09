import os
import sys
import torch
import argparse
import numpy as np
import random
import os.path as osp
from torch.backends import cudnn

from reid.utils.logging import Logger
from reid.data import build_data
from reid.engine import evaluate
from reid.models.msinet import msinet_x1_0
from reid.utils.serialization import copy_state_dict

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'test_log.txt'))
    print('Running evaluation with:{}'.format(args))

    _, test_loader, num_query, num_classes = build_data(args)
    model = msinet_x1_0(args, num_classes)
    model = torch.nn.DataParallel(model).cuda()
    
    if args.model_path == '':
        raise ValueError("Please specify the path to the trained model using --model-path")
    
    print(f'Loading model from {args.model_path}')
    copy_state_dict(torch.load(args.model_path), model)

    evaluate(args, model, test_loader, num_query, remove_cam=(args.target_dataset != 'vehicleid'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--source-dataset', type=str, default='market1501')
    parser.add_argument('-dt', '--target-dataset', type=str, default='none')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--num-instance', type=int, default=4)


    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--genotypes', type=str, default='msmt')
    
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--sam-mode', type=str, default='none')
    parser.add_argument('--sam-ratio', type=float, default=2.0)
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=100)
    
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--logs-dir', type=str, default='./logs')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()
    main(args)

from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch

from gcn_utils import load_data, accuracy
from model_wrapper import GCN_wrapper, GCN_uncertainty_wrapper

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--num_iter', type=int, default=200, help='number of epochs for training')
parser.add_argument('--num_train', type=int, default=140, help='number of nodes for training')
parser.add_argument('--num_val', type=int, default=300, help='number of nodes for validation')
parser.add_argument('--num_test', type=int, default=1000, help='number of nodes for testing')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=66, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data

data, idx_train, idx_val, idx_test = load_data(args,task="nc")

model = GCN_wrapper(data, args)
model.fit(idx_train)
model_posthoc = GCN_uncertainty_wrapper(model, idx_train)
predictions, ci = model_posthoc.ci_construction(idx_test, idx_test)
acc_test = accuracy(predictions, data["labels"][idx_test])
print("Test set results:", "accuracy= {:.4f}".format(acc_test.item()))
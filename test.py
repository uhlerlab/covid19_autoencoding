import neural_model as nm
import torch
import numpy as np
from copy import deepcopy
import data_loader as dl
import options_parser as op
import pandas as pd
import random
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle as p
from itertools import permutations
from random import sample


def get_embedding(net, x):
    o = deepcopy(x)
    for idx, layer in enumerate(net.net):
        o = layer(o)
        if idx == 1:
            break
    return o


# Checks the test accuracy of trained network
def main(args):
    SEED = 17
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    net = nm.Net()
    path = './trained_model_best.pth'
    d = torch.load(path)
    net.load_state_dict(d['state_dict'])

    net.eval()
    net.cuda()

    train_loader, test_loader = dl.get_data(args.data)
    val_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(test_loader):
        inputs = torch.autograd.Variable(batch).cuda()
        with torch.no_grad():
            output = net(inputs)
        loss = criterion(output, inputs)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(test_loader.dataset)
    print(val_loss)

if __name__ == "__main__":
    args = op.setup_options()
    main(args)

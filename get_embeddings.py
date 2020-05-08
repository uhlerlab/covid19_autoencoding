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
import pandas
import cmapPy
from cmapPy.pandasGEXpress.write_gctx import write
from cmapPy.pandasGEXpress.parse import parse

def get_embedding(net, x):
    o = deepcopy(x)
    for idx, layer in enumerate(net.net):
        o = layer(o)
        if idx == 0:
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

    pairs = dl.get_all_pairs(args.data)
    embeddings = {}
    for idx, p in enumerate(pairs):
        if idx % 10000 == 0 :
            print(idx)
        with torch.no_grad():
            input = torch.from_numpy(p[1]).view(1, -1)
            embedding = get_embedding(net, input)
            embeddings[p[0]] = embedding.data.numpy()[0]

    df = pandas.DataFrame(data=embeddings)
    out = cmapPy.pandasGEXpress.GCToo.GCToo(df)
    write(out, 'embeddings')

if __name__ == "__main__":
    args = op.setup_options()
    main(args)

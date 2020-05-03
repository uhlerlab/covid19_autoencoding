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
import umap
import colorsys
import neural_model as nm
import torch


def get_embedding(net, x):
    o = deepcopy(x)
    for idx, layer in enumerate(net.net):
        o = layer(o)
        if idx == 0:
            break
    return o


def norm(vec):
    return np.sqrt(np.sum(np.power(vec,2)))


def get_correlation(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def main(args):
    SEED = 1717
    np.random.seed(SEED)
    random.seed(SEED)
    net = nm.Net()
    path = './trained_model_best.pth'
    d = torch.load(path)
    net.load_state_dict(d['state_dict'])
    net.double()
    net.eval()
    net.cuda()

    pairs = dl.get_all_pairs(args.data)
    embeddings = []
    cell_types = {}
    idx_to_cell = []
    cell_1 = 'A549'
    cell_2 = 'MCF7'
    fda_approved = set([])

    with open('fda_approved.txt', 'r') as f:
        for line in f:
            fda_approved.add(line.strip())

    pert_cell_map = p.load(open('pert_cell_map.p', 'rb'))
    pert_dose_map = p.load(open('pert_dose_map.p', 'rb'))
    embedding_map = {}
    # Annoyingly pert id is actually pert name in the dataframe
    for i in range(len(pairs)):
        cell_type = pert_cell_map[pairs[i][0]]

        pert_id = pert_dose_map[pairs[i][0]][2]
        pert_type = pert_dose_map[pairs[i][0]][3]
        rna_plate = pert_dose_map[pairs[i][0]][4]
        rna_well = pert_dose_map[pairs[i][0]][5]
        if pert_dose_map[pairs[i][0]][0] < 0:
            continue
        if cell_type != cell_1 and cell_type != cell_2:
            continue
        if pert_id not in fda_approved:
            continue
        if pert_id != 'DMSO' and pert_id != 'vorinostat':
            continue

        cell_type = cell_type + "_" + pert_id
        #if cell_type in cell_types and len(cell_types[cell_type]) > 50:
        #    continue

        idx_to_cell.append(cell_type)
        if cell_type in cell_types:
            cell_types[cell_type].append(i)
        else:
            cell_types[cell_type] = [i]
        input = torch.from_numpy(pairs[i][1]).double().cuda()
        input = input.view(1, -1)
        # Uncomment to use original embedding
        #embedding = pairs[i][1].reshape(1, -1)

        # Recomment when using original embedding
        embedding = get_embedding(net, input)
        embedding = embedding.cpu().data.numpy()

        embedding_map[i] = embedding
        embeddings.append(embedding)

    print(sorted(cell_types.keys()))
    print(len(sorted(cell_types.keys())))
    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)

    means = {}
    for key in cell_types:
        points = np.array([embedding_map[i] for i in cell_types[key]])
        means[key] = np.mean(points, axis=0).reshape(-1)

    print("CORRELATIONS:")
    print(get_correlation(means[cell_1 + '_vorinostat'] \
                          - means[cell_1 + '_DMSO'],
                          means[cell_2 + '_vorinostat'] \
                          - means[cell_2 + '_DMSO']))


    reducer = umap.UMAP()
    embedding = reducer.fit_transform(embeddings)
    print("Shape after transform: ", embeddings.shape)
    cell_keys = sorted(cell_types.keys())
    color_map = {cell_keys[i]: i for i in range(len(cell_keys))}
    color_lvl = 8

    rgb = np.array(list(permutations(range(0,256,color_lvl),3)))/255
    colors = sample(list(rgb), len(cell_keys))
    seen = set([])
    group_by_color = {}

    for idx, e in enumerate(embedding):
        cell_type = idx_to_cell[idx]
        if cell_type in group_by_color:
            group_by_color[cell_type].append(e)
        else:
            group_by_color[cell_type] = [e]

    for key in group_by_color:
        points = np.array(group_by_color[key])
        cell_type = key
        plt.plot(points[:, 0], points[:, 1], 'o',
                 color=colors[color_map[cell_type]],
                 label=cell_type,
                 alpha=.5)

    plt.legend(bbox_to_anchor=(1.05, 1), ncol=10)
    plt.savefig('tmp.png', bbox_inches='tight')


if __name__ == "__main__":
    args = op.setup_options()
    main(args)

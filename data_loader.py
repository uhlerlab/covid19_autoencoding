from cmapPy.pandasGEXpress.parse import parse
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader


# Returns Train/Test Loaders from provided gctx data
# path := path to the gctx dataset
def get_data(path):
    shuffled_pairs = get_all_pairs(path)

    split_idx = int(len(shuffled_pairs) * .9)
    print("Number of Training Pairs: ", split_idx)
    print("Number of Test Pairs: ", len(shuffled_pairs) - split_idx)
    train_pairs = shuffled_pairs[:split_idx]
    test_pairs = shuffled_pairs[split_idx:]

    train_set = GeneVecs(train_pairs)
    test_set = GeneVecs(test_pairs)

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=2, shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=2, shuffle=False,
                             pin_memory=True)
    return train_loader, test_loader

# Retrieves all gene expression vectors from gctx dataset
def get_all_pairs(path):
    g_df = parse(path)
    df = g_df.data_df
    keys = list(df.keys())
    vectors = df.values.transpose()

    # To use unit norm scaling uncomment below
    #norm = np.sqrt(np.sum(np.power(vectors, 2), axis=1)).reshape(-1,1)
    #vectors = vectors / norm # unit norm inputs

    # To use z-score scaling uncomment below
    #means = np.mean(vectors, axis=1).reshape(-1, 1)
    #std = np.std(vectors, axis=1).reshape(-1, 1)
    #vectors = (vectors - means) / std

    pairs = list(zip(keys, vectors))
    idxs = np.random.permutation(np.arange(len(pairs)))
    shuffled_pairs = [pairs[idxs[i]] for i in idxs]
    return shuffled_pairs


# Stores gene expression vectors as pytorch dataset
class GeneVecs(Dataset):

    # Data comes in as a list
    def __init__(self, train_pairs):
        self.train_pairs = train_pairs

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        sample = self.train_pairs[idx][1]
        return sample

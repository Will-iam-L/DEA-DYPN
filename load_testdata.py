import pickle
import torch


def load_testdata(filename=None, offset=0, num_samples=128):

    with open(filename, 'rb') as f:
        data = pickle.load(f)
        data = [
            {
                'loc': torch.FloatTensor(loc),
                'prize': torch.FloatTensor(prize),
                'depot': torch.FloatTensor(depot),
                'max_length': torch.tensor(max_length)
            }
            for depot, loc, prize, max_length in (data[offset:offset + num_samples])
        ]
    return data
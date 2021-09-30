import numpy as np
import torch
class SplitedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split_idxs):
        self.dataset = dataset
        self.idxs = split_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

def split_dataset(dataset, seed, ratio):
    idx_list = np.arange(len(dataset))
    print(idx_list)
    
    np.random.seed(seed)
    np.random.shuffle(idx_list)
    train_len = round(len(idx_list) * ratio[0] / 100)
    print(train_len)
    
    valid_len = train_len + round(len(idx_list) * ratio[1] / 100)
    train_dataset = SplitedDataset(dataset, idx_list[:train_len])
    print(train_dataset)

    valid_dataset = SplitedDataset(dataset, idx_list[train_len:valid_len])
    test_dataset = SplitedDataset(dataset, idx_list[valid_len:])

    return {
        "train": train_dataset if len(train_dataset) > 0 else None, 
        "valid": valid_dataset if len(valid_dataset) > 0 else None, 
        "test": test_dataset if len(test_dataset) > 0 else None,
    }
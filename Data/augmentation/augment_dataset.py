import torch

from copy import deepcopy
from typing import Sequence
from torch.utils.data import Dataset
from .augment import DataAugment


class AugmentDataset(Dataset):
    """
    A augmented dataset that used to combine with origional training
    dataset to implement offline data augmentation.

    Parameters
    ----------
    dataset : Dataset
        The whole dataset will be used.
    indices : Sequence[int]
        The selected indices for the data used to do augmentation.
    involved_keys : list
        The key specify in each sample, which item involved for
        the data augmentation.
    w_array : int
        The width of the array used to record neural data.
    h_array : int
        The height of the array used to record neural data.
    n_bands : int
        Number of frequency bands of the neural data.
    methods : list
        The augmentation methods want to be used. It can be
        'jittering', 'scaling', 'permutation', 'rotation',
        'mixup'.

    Notes
    -----
    For now, do not fully support 'mixup'.
    """
    def __init__(self, dataset: Dataset, indices: Sequence[int],
                 involved_keys: list, w_array: int, h_array: int,
                 n_bands: int, methods: list):
        # To generate offline augmented dataset, the probability is
        # 100%.
        self.augmentor = DataAugment(
            1.0, w_array, h_array, n_bands, methods
        )

        # The dict list for all augmentation data.
        self.dict_list = []

        # Get the selected data to do augmentation.
        for i in indices:
            # Obtain the sample.
            sample = dataset[i]
            # Push to the storage list.
            self.dict_list.append(deepcopy(sample))

            # The object for the data augmentation.
            X_ = self.dict_list[-1]['inputs']
            # The objects which affected by the augmentation.
            Y_ = torch.tensor([], device=X_.device)
            # This used to recorde the index of each affected object.
            item_indices = [0]
            for key in involved_keys:
                value = self.dict_list[-1][key]
                item_indices.append(value.size(-1) + item_indices[-1])
                Y_ = torch.cat([Y_, value], dim=-1)

            # Do augmentation.
            X, Y = self.augmentor(X_, Y_)

            self.dict_list[-1]['inputs'] = X
            # uncat the Y.
            for j, key in enumerate(involved_keys):
                beg, end = item_indices[j], item_indices[j + 1]
                self.dict_list[-1][key] = Y[:, beg:end]

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, idx):
        return self.dict_list[idx]

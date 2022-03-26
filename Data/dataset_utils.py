import numpy as np
import copy

from torch.utils.data import Dataset, DataLoader, random_split


def split_dataset(dataset: Dataset, nfold, ktest, batch_size, drop_last=False):
    len_dts = len(dataset)
    len_tst = int(len_dts * ktest)
    len_val = 0 if nfold <= 1 else (len_dts - len_tst) // nfold
    len_tra = len_dts - (len_tst + len_val)

    # The list of the loaders contained all folds of the data.
    tra_loaders, val_loaders = [], []
    # The indices of the whold dataset.
    indices = np.linspace(0, len_dts - 1, len_dts, dtype=int)
    # Random split the dataset to training, validation, and test.
    tra_dts, val_dts, tst_dts = random_split(
        dataset, [len_tra, len_val, len_tst]
    )

    # 1. Get the test dataset.
    # The test data always at the last part for time sequence.
    tst_dts.indices = indices[-len_tst:]
    tst_loader = DataLoader(tst_dts, batch_size=batch_size)

    # 2. Get nfold of training and validation set.
    n = nfold if nfold > 0 else 1
    for i in range(n):
        # The indices of the validation dataset.
        beg = len_val * i
        end = len_val * (i + 1)
        val_dts.indices = indices[beg:end]
        # Push to the validation dataloader list.
        if val_dts.indices:
            val_loaders.append(DataLoader(
                copy.deepcopy(val_dts), batch_size, True, drop_last=drop_last)
            )

        # The indices of the training dataset.
        tra_dts.indices = np.concatenate((indices[:beg],
                                          indices[end:-len_tst]))
        # Push to the train dataloader list.
        tra_loaders.append(DataLoader(
            copy.deepcopy(tra_dts), batch_size, True, drop_last=drop_last)
        )
    # END of N_FOLD
    print('\nThe data set size of training, validation '
          f'and test: ({len_tra}, {len_val}, {len_tst}).\n')
    return tra_loaders, val_loaders, tst_loader

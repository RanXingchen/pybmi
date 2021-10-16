import os
import numpy as np
import torch

from torch.utils.data import Dataset
from pybmi.data import HTKFile


class HTKDataset(Dataset):
    """
    A pytorch-based dataset that loading HTK format data and
    used for Speech BCI.

    This dataset is iteratable and the loaded data type is
    pytorch tensor.

    Parameters
    ----------
    data_path : str
        Root path of the speech dataset. Under this path, it
        should contain subfolders which have the input data and
        target data.
    inp_type : list
        The type list of the input data. This contains by strings,
        where these string names should be same as the name of the
        input subfolder and the extension of the file name. For
        example, INP_TYPE=['ecog', 'mea'] will load the ecog data
        and mea data to form the input.
    trg_type : list
        The type list of the target data. This contains by strings,
        where these string names should be same as the name of the
        target subfolder and the extension of the file name. For
        example, TRG_TYPE=['mel', 'pitch'] will load the mel data
        and pitch data as the target.
    lat_type : list, optional
        The type list of the latent data. This contains by strings,
        where these string names should be same as the name of the
        latent subfolder and the extension of the file name. For
        example, LAT_TYPE=['ema'] will load the ema data as the
        latent. Default: []
    add_type : str, optional
        The additional type of data if required, then can be loaded
        by specified the name to the ADD_TYPE. Note that only one
        type can be loaded to the additional data. Default: ''.
    dtype : torch.dtype, optional
        The data type of the loading data.
    device : str, optional.
        The desired device of returned dataset. Default: 'cuda'.

    Attribute
    ---------
    inp_size : dict
        A dict contains each type of the feature size of input data.
        The key of the dict is same with the INP_TYPE.
    trg_size : dict
        A dict contains each type of the feature size of target data.
        The key of the dict is smae with the TRG_TYPE.
    lat_size : dict
        A dict contains each type of the feature size of latent data.
        The key of the dict is same with the LAT_TYPE.

    Returns
    -------
    __getitem__ method receive an index as input and return a dict
    that contained the index of data.
    If no latent and additional data:
        {
            'inputs': inp_data[index],
            'target': trg_data[index],
            'fnames': fnames[index]
        }
    If contain latent data or additional data:
        {
            'inputs': inp_data[index],
            'target': trg_data[index],
            'latent' / add_type: lat_data[index] / add_data[index],
            'fnames': fnames[index]
        }
    If both contain latent and additional data:
        {
            'inputs': inp_data[index],
            'target': trg_data[index],
            'latent': lat_data[index],
            add_type: add_data[index],
            'fnames': fnames[index]
        }
    """

    def __init__(self, data_path: str, inp_type: list, trg_type: list,
                 lat_type: list = [], add_type: str = '',
                 dtype: torch.dtype = torch.float,
                 device: str = 'cuda'):
        self.dtype = dtype
        self.device = device

        # Get file names of input data and target data.
        self.fnames = self._get_file_names(
            data_path, [*inp_type, *trg_type]
        )

        # * Load the whole dataset

        # Input and target data are required to be loaded.
        self.inp_data, self.inp_size = self._load_files(
            data_path, inp_type
        )
        self.trg_data, self.trg_size = self._load_files(
            data_path, trg_type
        )

        # Latent data only useful when the models are cases like
        # encoder-decoder framework. User can choice to not load
        # this type of data.
        self.lat_type = lat_type
        if lat_type:
            self.lat_data, self.lat_size = self._load_files(
                data_path, lat_type
            )
        
        # Additional data normally used to assistance. 
        # For example: Silence data used for speech dataset when
        # computing the MCD, it's not involved in the models
        # construction.
        self.add_type = add_type
        if add_type.strip() != '':
            self.add_data, self.add_size = self._load_files(
                data_path, [add_type]
            )

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # The basic data dictionary.
        inp = torch.tensor(self.inp_data[idx], dtype=self.dtype,
                           device=self.device)
        trg = torch.tensor(self.trg_data[idx], dtype=self.dtype,
                           device=self.device)
        data_dict = {
            'inputs': inp,
            'target': trg,
            'fnames': self.fnames[idx]
        }
        # Check if other items required.
        if self.lat_type:
            lat = torch.tensor(self.lat_data[idx], dtype=self.dtype,
                               device=self.device)
            data_dict['latent'] = lat
        if self.add_type.strip() != '':
            add = torch.tensor(self.add_data[idx], dtype=self.dtype,
                               device=self.device)
            data_dict[self.add_type] = add
        return data_dict

    def _get_file_names(self, root_path: str,
                        subfolders: list) -> list:
        """
        Get all the file names under the subfolders.

        Parameters
        ----------
        root_path : str
            The root path which contain all subfolders.
        subfolders : list
            A list of subfolders. All files under the all subfolders
            will be obtained.

        Retures
        -------
        f : list
            The sorted list that contain all files of the specified
            folders.
        """
        f = [sorted(os.listdir(os.path.join(root_path, subfolder)))
             for subfolder in subfolders]

        n_subfolder = len(f)

        # Check if the number of file names in each subfolder are
        # equal.
        len_idx0 = len(f[0])
        for i in range(1, n_subfolder):
            assert len_idx0 == len(f[i]), \
                "Wrong number of files occored!"
        # Check if the names are same for each subfolder
        fnames = []
        for i in range(len_idx0):
            fnames.append(os.path.splitext(f[0][i])[0])
            for j in range(1, n_subfolder):
                namej, _ = os.path.splitext(f[j][i])
                assert fnames[i] == namej, \
                    "Wrong name of files occored!"

        # Check passed, get the file names list.
        return fnames

    def _load_files(self, root_path: str, subfolders: list):
        """
        Loading files from the specified path.

        Parameters
        ----------
        root_path : str
            The root path which contain the subfolder.
        subfolder : list
            The folders that contain data will be loaded.
            This list should have same length with FNAMES.
        """
        data_lists = []
        size_dict = {}
        # Load data from each subfolder.
        for subfolder in subfolders:
            p = os.path.join(root_path, subfolder)
            data_lists.append(
                [np.stack(HTKFile().load(
                    os.path.join(p, f + '.' + subfolder)))
                 for f in self.fnames]
            )
            # Record the size each data under the subfolder.
            size_dict[subfolder] = data_lists[-1][0].shape[-1]

        data_list = []
        # Integrate all subfolders data into one array.
        for zipped in zip(*data_lists):
            data_list.append(np.concatenate(zipped, axis=-1))
        return data_list, size_dict

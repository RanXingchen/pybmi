import matlab
import matlab.engine
import numpy as np
import torch
import os
import tkinter
import re
from decimal import Decimal

from dateutil.parser import parse
from typing import List


def check_params(candidate, choices: List, arg_name: str):
    """
    Check validation of input argument from a finite set
    of choices.

    Parameters
    ----------
    candidate : int or float or str
        The candidate parameter value.
    choices : list
        A list of all legal parameter values.
    arg_name : str
        The name of the parameter.

    Returns
    -------
    selected : int or float or str
        The validated input parameter value.

    Examples
    --------
    >>> from PYTHON.Utils.utils import check_param
    >>> ret = check_param('KF', ['LR', 'KF'], 'decoder')
    """

    str_choices = ''.join('\'' + str(i) + '\', '
                          for i in choices)
    str_choices = str_choices[:-2] + '.'

    assert candidate in choices, \
        "'" + str(candidate) + '\' is not a valid value '\
        'for the \'' + arg_name + '\' argument. '\
        'Valid values are: ' + str_choices

    selected = candidate
    return selected


def check_file(file_path, title='',
               file_types=(("all files", "*.*"))) -> str:
    if not os.path.exists(file_path):
        # Hidden the main window of Tk.
        tkinter.Tk().withdraw()
        # Popup the Open File UI. Get the file name and path.
        file_path = tkinter.filedialog.askopenfilename(
            title=title,
            filetypes=file_types
        )
    return file_path


def npc_remove(x, npc=b'\x00', code='utf8'):
    """
    Decoding input bytes x with code and remove
    non-printable character from it.

    Parameters
    ----------
    x : bytes
        The input data need to remove some non-printable
        character.
    npc : bytes, optional
        The target non-printable character will be removed.
        Default: b'\x00'.
    code : str, optional
        The performed decode type, default vaule is 'utf8'.

    Returns
    -------
    The decoded data of x which removed the NPCs.
    """
    return x.decode(code).strip(npc.decode(code))


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    Parameters
    ----------
    string: str
        String to check for date.
    fuzzy: bool, optional
        Ignore unknown tokens in string if True

    Returns
    -------
    Return True if the string can be interpreted as date,
    otherwise return False.
    """

    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def elapsed_time(start_time: int, end_time: int):
    """
    Computing the running time from the start to the end.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def update_dict(dict1: dict, dict2: dict):
    """
    Update the target dict by reference dict when both variable have
    the same keys.

    Parameters
    ----------
    dict1 : dict
        The target dict.
    dict2 : dict
        The reference dict.

    Returns
    -------
    new_dict : dict
        The updated dictionary according to the reference.
    """
    d = dict1.copy()

    for key in dict1.keys():
        if key in dict2.keys():
            d[key] = dict2[key]
    return d


def find_padding_index(data: torch.Tensor, padding_value):
    """
    Find the padding index from data by the padding_value.
    """
    if padding_value != padding_value:
        # Padding value is nan.
        iPad = torch.isnan(data)
    elif isinstance(padding_value, int) or \
            isinstance(padding_value, float):
        iPad = data == padding_value
    elif padding_value is None:
        # No padding value.
        iPad = torch.zeros((data.shape[0], data.shape[1]),
                           dtype=bool, device=data.device)
    else:
        raise ValueError("Unsupported type of padding value.")
    return iPad


def t2s(timestr: str, decimal_place=3) -> Decimal:
    """
    Convert text format time to second.
    """
    unit = str(1 / pow(10, decimal_place))

    if not re.search(':', timestr):
        return Decimal(timestr)

    time = re.split(':', timestr)
    sec = Decimal('0')
    for n, i in enumerate(range(len(time), 0, -1)):
        # Check if the time string has decimal
        if re.search(r'\.', time[i - 1]):
            time_ = re.split(r'\.', time[i - 1])
            sec += (Decimal(time_[0]) + Decimal(time_[1]) *
                    Decimal(unit)) * pow(60, n)
        else:
            sec += Decimal(time[i - 1]) * pow(60, n)
    return sec


class Array2mat():
    """
    Convert numpy ndarray to matlab matrix.
    For now, it's only support 1D or 2D numpy array or
    python list.

    Parameters
    ----------
    x : list or ndarray or tensor
        The numpy ndarray that contain values, which data
        type must be
        float. If not, it will be cast to float automaticly.
    """
    def __init__(self):
        self.eng = matlab.engine.start_matlab()

    def __call__(self, x):
        if type(x) == list:
            x = np.array(x)
        elif type(x) == torch.Tensor:
            x = x.cpu().numpy()
        # Check the type of x is validate.
        assert type(x) == np.ndarray, \
            'Wrong type of input: ' + str(type(x)) + '.'
        # Make sure the dimension of x is not greater than 2
        D = x.ndim
        assert D <= 2, \
            f'{D}D array is not supported for now!'
        # Convert the vector to row vectors.
        if D == 1 or x.shape[1] == 1:
            x = np.reshape(x, (1, -1))
        if x.dtype != np.float:
            x = x.astype(np.float)

        # Transform the array.
        mat = self.eng.cell2mat(
            np.reshape(x, -1, order='F').tolist()
        )
        # Get the right shape of mat
        mat = self.eng.reshape(mat, x.shape[0], x.shape[1])
        return mat

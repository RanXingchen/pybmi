from dateutil.parser import parse


def check_params(candidate, choices, arg_name):
    """
    Check validation of input argument from a finite set of choices.

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
    >>> ret = check_param('KF', ['LR', 'KF', 'RNN'], 'decoder')
    """

    str_choices = ''.join('\'' + str(i) + '\', ' for i in choices)
    str_choices = str_choices[:-2] + '.'

    assert candidate in choices, \
        "'" + str(candidate) + '\' is not a valid value for the \'' + \
        arg_name + '\' argument. Valid values are: ' + str_choices

    selected = candidate
    return selected


def npc_remove(x, npc=b'\x00', code='utf8'):
    """
    Decoding input bytes x with code and remove
    non-printable character from it.

    Parameters
    ----------
    x : bytes
        The input data need to remove some non-printable character.
    npc : bytes, optional
        The target non-printable character will be removed. Default: b'\x00'.
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

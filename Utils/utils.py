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

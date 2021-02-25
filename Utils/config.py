import configparser


class Config():
    """
    Read configuration file from sepecified path.

    Parameters
    ----------
    filepath : str
        The path direction to a configuration file, which contain
        Sections, Keys and Items.
    defaults : dict
        A dictionary that contaion the default parameters for the task.
    verbose : bool, optional
        To print the readed config parameters or not. Default: False.
    """
    def __init__(self, filepath, defaults, verbose=False):
        self.parser = configparser.ConfigParser()
        self.config = defaults

        try:
            self.parser.read(filepath)
        except Exception as e:
            print(e)

        self.check_config()

        if verbose:
            self.print()

    def __getitem__(self, key):
        return self.config[key]

    def check_config(self):
        for sec in self.config.keys():
            # Make sure the readed section satisfy the default.
            if sec in self.parser:
                for key, value in self.config[sec].items():
                    # Check the key and value both contained by
                    # the file and default.
                    if key in self.parser[sec] and self.parser[sec][key]:
                        # Overwrite the value of cofig
                        if type(value) == str:
                            self.config[sec][key] = self.parser[sec][key]
                        else:
                            self.config[sec][key] = eval(self.parser[sec][key])

    def print(self):
        for section_key, section_value in self.config.items():
            for key, value in section_value.items():
                print('[''%s'']: %s: %s' % (section_key, key, value))

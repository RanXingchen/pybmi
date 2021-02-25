import sys
import os


class Logger():
    """
    Save the terminal printing content to the log file.


    Paramters
    ---------
    path : str
        The path to save the log file.
    name : str, optional
        Name of the log file.
    """
    def __init__(self, path, name='details.log'):
        self.terminal = sys.stdout
        assert os.path.exists(path), "The log writing path does not exist!"
        self.log = open(os.path.join(path, name), 'a', encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    sys.stdout = Logger('A:\\Documents', 'test.log')
    print("This is a test message.")
    print("--")
    print(":;;;")
    print("")
    print("我的达利园软面包")
    print("zzzzz")

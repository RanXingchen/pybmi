import os
import numpy as np
import array
import time

from Utils.utils import npc_remove


class BMIReader():
    """
    Reading the BMI format file. Usually, BMI file contain a header and data.

    Parameters
    ----------
    path : str
        The path of bmi file which will be readed.
    report : bool, optional
        To print a log on the terminal which contain the basic
        information of the readed BMI files, e.g., the header, file size,
        file version etc.
    """
    def __init__(self, path, report=False):
        # Supported file version.
        self.version_list = ['v1.0', 'v1.1', 'v1.2']
        self.report = report

        assert os.path.exists(path), \
            'The provided file \'' + path + '\' does not exist!'

        # Initialize the file information.
        self.file_size = os.path.getsize(path)
        self.file_path, self.file_name = os.path.split(path)
        self.header = {
            'version':      'Unknown',
            'subject':      'Unknown',
            'experimenter': 'Unknown',
            'paradigm':     'Unknown',
            'date':         'Missing',
            'comment':      '',
            'lbl_dim':      0,
            'spk_dim':      0,
            'kin_dim':      0,
            'fs':           0
        }

    def read(self, columns=None):
        """
        Parameters
        ----------
        columns : int, optional
            In some early version of BMI file, we didn't provide sufficient
            information to calculate the correct columns of BMI data. In such
            case, a provided columns can help reading normally.
        """
        tic = time.time()
        # Reading BMI files.
        with open(os.path.join(self.file_path, self.file_name), 'rb') as f:
            # Checking the possible version of the file.
            version = npc_remove(f.read(32), npc=b'\x00')

            # Process different version of BMI files.
            if version.lower == self.version_list[0]:
                print("Not implement yet.")
            elif version.lower == self.version_list[1]:
                print("Not implement yet.")
            elif version.lower == self.version_list[2]:
                print("Not implement yet.")
            else:
                # Unsupported file version. Perform default reading process.
                data, data_points = self._default_reading(f, columns)
                # TODO: convert the unsupported file version to latest version.

            # Calculate the recording time duration.
            mins = (data_points / self.header['fs']) // 60
            secs = (data_points / self.header['fs']) - mins * 60
        toc = time.time()

        if self.report:
            self._report_info(data_points, (mins, secs), tic, toc)
        return data

    def _default_reading(self, f, columns):
        f.seek(0, 0)

        # Reading file header in default way.
        self.header['subject'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['experimenter'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['paradigm'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['comment'] = npc_remove(f.read(256), npc=b'\x00')
        self.header['fs'] = int.from_bytes(f.read(1), byteorder='big')
        self.header['spk_dim'] = int.from_bytes(f.read(1), byteorder='big')
        self.header['kin_dim'] = int.from_bytes(f.read(1), byteorder='big')
        self.header['lbl_dim'] = int.from_bytes(f.read(1), byteorder='big')

        # Check the validation of sampling rate of BMI.
        if self.header['fs'] <= 0:
            print(
                "BMIReader::WARNING: sampling rate of BMI = "
                f"{self.header['fs']}. This seems uncorrect,"
                " set sampling rate to default value: 100."
            )
            self.header['fs'] = 100

        # Compute the data size, which equal to file_size - header_size
        data_size = self.file_size - f.tell()
        # The data columns specified how many data displayed in each row.
        # By default, in each row, there should contain:
        # [time stamp, label, moving_x, moving_y, target_x, target_y]
        data_cols = 1 + self.header['spk_dim'] + \
            self.header['kin_dim'] * 2 + self.header['lbl_dim'] \
            if columns is None else columns

        data_points = data_size // (8 * data_cols)
        assert data_points * 8 * data_cols == data_size, \
            'Reading BMI file \'' + \
            os.path.join(self.file_path, self.file_name) + '\' failed.' \
            f'The data size is not divisible by data columns {data_cols}.'

        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, data_cols))
        return data, data_points

    def _report_info(self, data_points, duration, t1, t2):
        print('*** FILE INFO **************************')
        print('File Path\t= ' + self.file_path)
        print('File Name\t= ' + self.file_name)
        print('File Extension\t= ' + self.file_name.split('.')[-1])
        print('File Version\t= ' + self.header['version'])
        print(f'Duration\t= {duration[0]:.0f}m {duration[1]:.0f}s')
        print('Data Points\t= ' + str(data_points))

        print('*** BASIC HEADER ***********************')
        print('Subject\t\t\t= ' + self.header['subject'])
        print('Experimenter\t\t= ' + self.header['experimenter'])
        print('Paradigm\t\t= ' + self.header['paradigm'])
        print('Date\t\t\t= ' + self.header['date'])
        print('Comment\t\t\t= ' + self.header['comment'])
        print('Sample Frequency\t= ' + str(self.header['fs']))
        print('Electrodes Read\t\t= ' + str(self.header['spk_dim']))
        print('Kinematic Dimension\t= ' + str(self.header['kin_dim']))

        print(f'The load time for BMI file was {(t2 - t1):.2f} seconds.')

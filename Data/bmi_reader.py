import os
import numpy as np
import array
import time
import tkinter

from pybmi.utils.utils import npc_remove, check_params
from tkinter import filedialog


class BMIReader():
    """
    Reading the BMI format file. Usually, BMI file contain a header and data.

    Parameters
    ----------
    filepath : str
        The path of bmi file which will be readed. If it is None, a UI will
        pop out to ask user select one file.
    report : bool, optional
        To print a log on the terminal which contain the basic
        information of the readed BMI files, e.g., the header, file size,
        file version etc.

    Attributes
    ----------
    header : dict
        Readed header information from specified BMI files. It contain
        the information of experiment like subject, paradigm, date,
        comment and so on, also, the information of data structure is
        included, such as the dimension of the label, the dimension of
        spike data, the dimension of kinematic data and sampling rate.
    rawdata : ndarray
        The data matrix readed from the BMI file, which didn't do any
        processing.
    data : ndarray
        Processed data matrix. e.g., binned, masked out some labels.
    stat : dict
        For the BCI paradigm, some statistics data can analyzed from
        the raw data, e.g., success rate, number of trials count. The
        stat dict contain these statistics information for supported
        BCI paradigm.

    Examples
    --------
    >>> path = 'MonkeyData\\201907031450.bmi'
    >>> reader = BMIReader(path, report=True)
    >>> bmi = reader.read(bin_size=0.1, columns=6, analyze=False)
    >>> reader.analysis(verbose=True)
    >>> masked, index = reader.filter((1, 4))
    """
    def __init__(self, filepath=None, report=False):
        # Supported file version.
        self.version_list = ['v1.0', 'v1.1', 'v1.2']
        self.report = report

        if filepath is None:
            # Hidden the main window of Tk.
            tkinter.Tk().withdraw()
            # Popup the Open File UI. Get the file name and path.
            filepath = filedialog.askopenfilename(
                title="Choose a behavior data file...",
                filetypes=(("BMI files", "*.bmi"), ("all files", "*.*"))
            )

        assert os.path.exists(filepath), \
            'The provided file \'' + filepath + '\' does not exist!'

        # Initialize the file information.
        self.file_size = os.path.getsize(filepath)
        self.path, self.name = os.path.split(filepath)
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

        # The BMI data
        self.rawdata = None
        # Binned BMI data
        self.data = None

        # Supported version of BMI data index.
        self.id_timestamp = {
            'Unknown': 0,
            'v1.0': 0,
            'v1.1': 0,
            'v1.2': 0
        }
        self.id_label = {
            'Unknown': 1,
            'v1.0': 1,
            'v1.1': 1,
            'v1.2': 1
        }
        # The finish label of one trial, first one represent success label,
        # the second one represent failure label.
        self.finish_labels = {
            'v1.2': [4, 5],
            'Unknown': [3, 4],
            # TODO: 不记得标签了，等有了对应版本的数据再加吧。
        }

        # Statistics dict which contain the Success Rate,
        # Total Number of Trials, Hold Time, Failed Time,
        # Average Time of Each Successed Trial.
        self.stat = {
            'num_trials':       0,
            'num_success':      0,
            'success_rate':     0.0,
            'hold_time':        0,
            'failed_time':      0,
            'time_to_target':   0.0,
        }

    def read(self, bin_size, fs_timestamp=30000, columns=None,
             method='sample', analyze=True, g=None, b=None):
        """
        Parameters
        ----------
        bin_size : float
            Bin size specified the time resolution of the processed BMI data.
            It should be equal to the bin size of corresponding neural data.
            Unit: seconds.
        fs_timestamp : int
            Sampling rate of time stamp. By default, the time stamp comes from
            recording NSP, which usually equal to 30000.
        columns : int, optional
            In some early version of BMI file, we didn't provide sufficient
            information to calculate the correct columns of BMI data. In such
            case, a provided columns can help reading normally.
        method : str, optional
            Method used to calculate the binned data. It can be 'sample' or
            'mean' either. 'sample' means only use the time of current sample
            point as the binned data; 'means' calculate average data from
            the time of last sample point to current sample point as the binned
            data.
        analyze : bool, optional
            Analyze the readed BMI raw data by the provided label and paradigm.
            After analysis, statistics parameter will be stored. For example,
            when paradigm is Center-Out, Success Rate, Total Number of Trials,
            Hold Time, Failed Time, Average Time of Each Successed Trial etc.
            will be calculated.
        g : int, optional
            A good label means subject's behavior successed satisfied
            the expriment. For example, hold enough time on the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified good label can help the analysis perform normally.
        b : int, optional
            A bad label means subject's behavior failed satisfied
            the expriment. For example, ran over the time to the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified bad label can help the analysis perform normally.
        """
        method = check_params(method, ['sample', 'mean'], 'method')

        tic = time.time()
        # Reading BMI files.
        with open(os.path.join(self.path, self.name), 'rb') as f:
            # Checking the possible version of the file.
            version = npc_remove(f.read(32), npc=b'\x00')

            # Process different version of BMI files.
            if version.lower() == self.version_list[0]:
                print("Not implement yet.")
                return 0
            elif version.lower() == self.version_list[1]:
                print("Not implement yet.")
                return 0
            elif version.lower() == self.version_list[2]:
                # v1.2
                self.rawdata, data_points = self._v1dot2_reading(f)
            else:
                # Unknown file version. Perform default reading process.
                self.rawdata, data_points = self._default_reading(f, columns)
                # TODO: convert the unsupported file version to latest version.
        toc = time.time()

        # Calculate the recording time duration.
        mins = (data_points / self.header['fs']) // 60
        secs = (data_points / self.header['fs']) - mins * 60

        # Analyze behavior if analyze==True
        if analyze:
            self.analysis(g, b)
            # Warning the label might missing if bin size greater
            # than hold time
            if self.stat['hold_time'] < bin_size * 1000:
                print(
                    'BMIReader::WARNING: Success label might be missing '
                    f'because bin size={bin_size * 1000}ms greater than '
                    f'hold time={self.stat["hold_time"]:.1f}ms.\n')

        # Bin data
        self.data = self._bin_data(bin_size, fs_timestamp, method=method)

        if self.report:
            self._report_info(data_points, (mins, secs), tic, toc, analyze)
        return self.data

    def analysis(self, g=None, b=None, verbose=False):
        """
        Trying to analyze the readed BMI file by the provided label
        information and paradigm. After analysis, statistics parameter
        will be calculated. For example, when paradigm is Center-Out,
        Success Rate, Total Number of Trials, Hold Time, Failed Time,
        Average Time of Each Successed Trial etc. will be calculated.

        Parameters
        ----------
        g : int, optional
            A good label means subject's behavior successed satisfied
            the expriment. For example, hold enough time on the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified good label can help the analysis perform normally.
        b : int, optional
            A bad label means subject's behavior failed satisfied
            the expriment. For example, ran over the time to the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified bad label can help the analysis perform normally.
        verbose : bool, optional
            Print the analyzed statistics on the terminal if verbose set
            to True, otherwise no print.
        """
        if self.rawdata is None:
            print('\nCan not do analyze until the BMI file readed!')
            return

        # Reset all stat
        self.reset_stat()
        version = self.header['version']

        if self.header['paradigm'].lower() in \
           ['center-out', 'center out', 'centerout']:
            # Center-out behavior analysis.

            # Default center-out success label and failure label by
            # different version.
            if g is None:
                g = self.finish_labels[version][0]
            if b is None:
                b = self.finish_labels[version][1]
            # All of the labels accorrding different version.
            labels = self.rawdata[:, self.id_label[version]]

            # Sum all success labels in data as the total hold time.
            self.stat['hold_time'] = sum(labels == g)

            # Count the statistics behavior of the data
            for i, label in enumerate(labels):
                # Skip the 1st time step
                if i == 0:
                    trial_start = i
                    continue

                if label != g and label != b and \
                        (labels[i - 1] == g or labels[i - 1] == b):
                    # The start of current trial.
                    trial_start = i
                elif label == g and labels[i - 1] != g:
                    # The first finish point of this trial is success label.
                    self.stat['num_success'] += 1
                    self.stat['num_trials'] += 1
                    self.stat['time_to_target'] += (i - trial_start)
                elif label == b and labels[i - 1] != b:
                    # The first finish point of this trial is failure label.
                    self.stat['num_trials'] += 1
                    self.stat['failed_time'] += (i - trial_start)

            # Finish checking the whole data one by one
            self.stat['success_rate'] = \
                self.stat['num_success'] / self.stat['num_trials'] * 100
            if self.stat['num_success'] != 0:
                self.stat['hold_time'] *= \
                    (1000 / self.header['fs']) / self.stat['num_success']
                self.stat['time_to_target'] *= (1 / self.header['fs']) /\
                    self.stat['num_success']
            else:
                self.stat['hold_time'] = float('NaN')
                self.stat['time_to_target'] = float('NaN')
            if self.stat['num_trials'] - self.stat['num_success'] != 0:
                self.stat['failed_time'] *= (1 / self.header['fs']) /\
                    (self.stat['num_trials'] - self.stat['num_success'])
            else:
                self.stat['failed_time'] = float('NaN')
        else:
            # Unsupported pardigm.
            print('\nBMIReader::WARNING: Unsupported paradigm ' +
                  self.header['paradigm'] + ', can\'t do analyze of it.')

        if verbose:
            self._print_stat()

    def filter(self, label, data=None):
        """
        Mask out specified trial type from data. The rest trials
        will concatenate one by one to form a new data.

        Parameters
        ----------
        label : {tuple, int}
            The labels of the kind of trial will be masked out. If the
            given labels are tuple, means that it specified one kind of
            trial with its start and end label number. If the given
            label is int, means that this kind of label will be removed
            in all trials, no matter what kind of trial (success, failure)
            it is.
        data : ndarray, optional
            Specify the data to perform the filter operation. If None, use
            readed binned data as the object.

        Returns
        -------
        masked : ndarray
            The data which masked out specified label.
        index : ndarray
            A vector stored the index of masked rows of BMI data.
        """
        # Get the version information
        version = self.header['version']
        # The index of the labels in each row.
        j = self.id_label[version]
        # If not specify the data, use the binned data from BMI file.
        if data is None:
            data = self.data
        if isinstance(label, tuple):
            # The tuple type label indicate trial start and
            # trial end of masked out trail type.
            l1, l2 = label

            index = np.ones(data.shape[0], dtype=np.bool)

            for i, row in enumerate(data):
                if i == 0:
                    trial_start, trial_end = 0, 0
                    continue

                if row[j] == l1 and data[i - 1, j] != l1:
                    trial_start = i
                elif row[j] == l2 and data[i + 1, j] != l2:
                    trial_end = i

                # End of one whole trial
                if trial_start < trial_end:
                    index[trial_start:trial_end + 1] = False
        elif isinstance(label, int):
            index = ~(data[:, j] == label)

        masked = data[index]
        return masked, index

    def _default_reading(self, f, columns):
        f.seek(0, 0)
        # Reading file header in default way.
        self._header(f, 'Unknown')

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
            os.path.join(self.path, self.name) + '\' failed.' \
            f'The data size is not divisible by data columns {data_cols}.'

        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, data_cols))
        return data, data_points

    def _v1dot2_reading(self, f):
        # Reading from file start.
        f.seek(0, 0)
        self._header(f, 'v1.2')
        # Check the validation of sampling rate of BMI.
        if self.header['fs'] <= 0:
            print(
                f"BMIReader::WARNING: file sampling rate: {self.header['fs']}."
                " This seems uncorrect, set it to default value: 100."
            )
            self.header['fs'] = 100

        # Compute the data size, which equal to file_size - header_size
        data_size = self.file_size - f.tell()
        # The data columns specified how many data displayed in each row.
        data_cols = 1 + self.header['lbl_dim'] + self.header['spk_dim'] + \
            self.header['kin_dim'] * 4 + 1 + 1

        data_points = data_size // (8 * data_cols)
        assert data_points * 8 * data_cols == data_size, \
            'Reading BMI file \'' + \
            os.path.join(self.path, self.name) + '\' failed.' \
            f'The data size is not divisible by data columns {data_cols}.'

        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, data_cols))
        return data, data_points

    def _bin_data(self, bin_size, fs_timestamp, method):
        """
        Bin the rawdata by bin size and check each bin size by time stamp.

        Parameters
        ----------
        bin_size : float
            Bin size specified the time resolution of the processed BMI data.
            It should be equal to the bin size of corresponding neural data.
            Unit: seconds.
        fs_timestamp : int
            Sampling rate of time stamp. By default, the time stamp comes from
            recording NSP, which usually equal to 30000.
        method : str
            Method used to calculate the binned data. It can be 'sample' or
            'mean' either. 'sample' means only use the time of current sample
            point as the binned data; 'means' calculate average data from
            the time of last sample point to current sample point as the binned
            data.
        """
        # Get the version information
        version = self.header['version']
        # The timestamp index
        t = self.id_timestamp[version]

        binned_data = []
        step_size_bmi = np.round(bin_size * self.header['fs'])
        step_size_tsp = bin_size * fs_timestamp
        last_bin_idx = 0
        for i, row in enumerate(self.rawdata):
            if i == 0:
                binned_data.append(row[np.newaxis, :])
                continue

            # When BMI data recording frequency up.
            if i <= last_bin_idx:
                continue

            # Time to get next bin.
            if (i - last_bin_idx) % step_size_bmi == 0:
                # Sometimes we think the recording of BMI file is not
                # reliable as NSP recording, so use NSP time stamp to
                # check if it is the accurate time of this bin.
                e = row[t] - self.rawdata[last_bin_idx, t] - step_size_tsp
                # Find the first index makes step >= step_size_tsp
                for i1 in range(i + 1, self.rawdata.shape[0]):
                    step = self.rawdata[i1, t] - self.rawdata[last_bin_idx, t]
                    # Stop condition
                    if step >= step_size_tsp:
                        break
                e1 = step - step_size_tsp
                # Find the first index makes step <= step_size_tsp
                for i2 in range(i - 1, last_bin_idx, -1):
                    step = self.rawdata[i2, t] - self.rawdata[last_bin_idx, t]
                    # Stop condition
                    if step <= step_size_tsp:
                        break
                e2 = step - step_size_tsp

                # Update the bin index
                if abs(e) <= abs(e1) and abs(e) <= abs(e2):
                    # The recording of BMI data right on time.
                    last_bin_idx = i
                elif abs(e1) < abs(e) and abs(e1) <= abs(e2):
                    # BMI data recording frequency up.
                    last_bin_idx = i1
                elif abs(e2) < abs(e) and abs(e2) < abs(e1):
                    # BMI data recording frequency down.
                    last_bin_idx = i2
                else:
                    print('BMIReader::WARNING: Debuging error!')

                if method == 'sample':
                    binned_data.append(self.rawdata[[last_bin_idx]])
                elif method == 'mean':
                    # TODO: add mean method.
                    print('not implement yet.')

        return np.concatenate(binned_data)

    def _report_info(self, data_points, duration, t1, t2, analyze=True):
        print('*** FILE INFO **************************')
        print('File Path\t= ' + self.path)
        print('File Name\t= ' + self.name)
        print('File Extension\t= ' + self.name.split('.')[-1])
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

        if analyze:
            self._print_stat()

        print(f'The load time for BMI file was {(t2 - t1):.2f} seconds.')

    def _print_stat(self):
        print('*** BEHAVIOR ANALYSIS ******************')
        print(f"Total Number of Trials\t\t= {self.stat['num_trials']}")
        print(f"Number of Success Trials\t= {self.stat['num_success']}")
        print(f"Success Rate\t\t\t= {self.stat['success_rate']:.2f}%")
        print(f"Hold Time\t\t\t= {self.stat['hold_time']:.0f}ms")
        print(f"Ran over failed time\t\t= {self.stat['failed_time']:.2f}s")
        print(f"Time to Target\t\t\t= {self.stat['time_to_target']:.2f}s")

    def _header(self, f, version):
        if version in self.version_list:
            self.header['version'] = npc_remove(f.read(32), npc=b'\x00')
            assert version == self.header['version'], \
                "Specified version mismatch readed version!"

        self.header['subject'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['experimenter'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['paradigm'] = npc_remove(f.read(32), npc=b'\x00')
        if version in self.version_list:
            self.header['date'] = npc_remove(f.read(32), npc=b'\x00')

        self.header['comment'] = npc_remove(f.read(256), npc=b'\x00')

        if version in self.version_list:
            self.header['countfirst'] = npc_remove(f.read(32), npc=b'\x00')
            self.header['lbl_dim'] = int.from_bytes(f.read(4), 'little')
            self.header['spk_dim'] = int.from_bytes(f.read(4), 'little')
            self.header['kin_dim'] = int.from_bytes(f.read(4), 'little')
            self.header['fs'] = int.from_bytes(f.read(4), 'little')
        else:
            self.header['fs'] = int.from_bytes(f.read(1), 'little')
            self.header['spk_dim'] = int.from_bytes(f.read(1), 'little')
            self.header['kin_dim'] = int.from_bytes(f.read(1), 'little')
            self.header['lbl_dim'] = int.from_bytes(f.read(1), 'little')

    def reset_stat(self):
        """
        Set the statistics dict to zero.
        """
        self.stat['num_trials'] = 0
        self.stat['num_success'] = 0
        self.stat['success_rate'] = 0
        self.stat['hold_time'] = 0
        self.stat['failed_time'] = 0
        self.stat['time_to_target'] = 0

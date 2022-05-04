import subprocess
import struct


def HCopy(conf, wav, htk):
    """
    Runs HTK HCopy program to calculate signal parameters.

    Parameters
    ----------
    conf : string
        Path to configuration file.
    wav : string
        Path to audio file.
    htk :string
        Path to output HTK file.

    Retures
    -------
    output : string
        Program output and possible errors. None if program didn't run.
    """
    output = None
    try:
        output = subprocess.check_output(['HCopy', '-C', conf, '-T', '1', wav,
                                          htk])
    except subprocess.CalledProcessError as e:
        print('EXC {}'.format(e))
    return output


class HTKFile:
    """
    Class to load binary HTK file.

    Details on the format can be found online in HTK Book chapter 5.7.1.

    Not everything is implemented 100%, but most features should be
    supported.

    Not implemented:
        CRC checking - files can have CRC, but it won't be checked for
        correctness

        VQ - Vector features are not implemented.
    """

    data = None
    nSamples = 0
    nFeatures = 0
    sampPeriod = 0
    basicKind = None
    qualifiers = None
    endian = '>'

    def load(self, filename):
        """
        Loads HTK file.

        Attributes
        ----------
        data :matrix
            Data contained in the file.
        nSamples : int
            Number of frames in the file.
        nFeatures : int
            Number of features per frame.
        sampPeriod : int
            Sample period in 100ns units (e.g. fs=16 KHz -> 625).
        basickind : string
            Basic feature kind saved in the file.
        qualifiers :string
            Feature options present in the file
        """
        with open(filename, 'rb') as f:
            header = f.read(12)
            self.nSamples, self.sampPeriod, sampSize, sampKind = \
                struct.unpack('>iihh', header)
            if self.nSamples < 0 or self.sampPeriod < 0 or sampSize < 0:
                self.endian = '<'
                self.nSamples, self.sampPeriod, sampSize, sampKind = \
                    struct.unpack(self.endian + 'iihh', header)
            basicParameter = sampKind & 0x3F

            if basicParameter == 0:
                self.basicKind = 'WAVEFORM'
            elif basicParameter == 1:
                self.basicKind = 'LPC'
            elif basicParameter == 2:
                self.basicKind = 'LPREFC'
            elif basicParameter == 3:
                self.basicKind = 'LPCEPSTRA'
            elif basicParameter == 4:
                self.basicKind = 'LPDELCEP'
            elif basicParameter == 5:
                self.basicKind = 'IREFC'
            elif basicParameter == 6:
                self.basicKind = 'MFCC'
            elif basicParameter == 7:
                self.basicKind = 'FBANK'
            elif basicParameter == 8:
                self.basicKind = 'MELSPEC'
            elif basicParameter == 9:
                self.basicKind = 'USER'
            elif basicParameter == 10:
                self.basicKind = 'DISCRETE'
            elif basicParameter == 11:
                self.basicKind = 'PLP'
            else:
                self.basicKind = 'ERROR'

            self.qualifiers = []
            if (sampKind & 0o100) != 0:
                self.qualifiers.append('E')
            if (sampKind & 0o200) != 0:
                self.qualifiers.append('N')
            if (sampKind & 0o400) != 0:
                self.qualifiers.append('D')
            if (sampKind & 0o1000) != 0:
                self.qualifiers.append('A')
            if (sampKind & 0o2000) != 0:
                self.qualifiers.append('C')
            if (sampKind & 0o4000) != 0:
                self.qualifiers.append('Z')
            if (sampKind & 0o10000) != 0:
                self.qualifiers.append('K')
            if (sampKind & 0o20000) != 0:
                self.qualifiers.append('0')
            if (sampKind & 0o40000) != 0:
                self.qualifiers.append('V')
            if (sampKind & 0o100000) != 0:
                self.qualifiers.append('T')

            if 'C' in self.qualifiers or 'V' in self.qualifiers or \
                self.basicKind == 'IREFC' or \
                    self.basicKind == 'WAVEFORM':
                self.nFeatures = sampSize // 2
            else:
                self.nFeatures = sampSize // 4

            if 'C' in self.qualifiers:
                self.nSamples -= 4

            if 'V' in self.qualifiers:
                raise NotImplementedError('VQ is not implemented')

            self.data = []
            if self.basicKind == 'IREFC' or self.basicKind == 'WAVEFORM':
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(
                            self.endian + 'h', s, v * 2)[0] / 32767.0
                        frame.append(val)
                    self.data.append(frame)
            elif 'C' in self.qualifiers:
                A = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    A.append(struct.unpack_from(
                        self.endian + 'f', s, x * 4)[0])
                B = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    B.append(struct.unpack_from(
                        self.endian + 'f', s, x * 4)[0])

                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        frame.append((struct.unpack_from(
                            self.endian + 'h', s, v * 2)[0] + B[v]) / A[v])
                    self.data.append(frame)
            else:
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(self.endian + 'f', s, v * 4)
                        frame.append(val[0])
                    self.data.append(frame)

            if 'K' in self.qualifiers:
                print('CRC checking not implemented...')

            return self.data

    def save(self, filename, data, sampPeriod, sampKind=9):
        """
        Save HTK files.

        The data format should be nSamples * nFeatures matrix.

        For doc see:
        https://labrosa.ee.columbia.edu/doc/HTKBook21/node58.html
        """
        with open(filename, 'wb') as f:
            nSamples, nFeatures = data.shape
            if sampKind == 9:
                sampSize = nFeatures * 4
            else:
                raise NotImplementedError('save as sampKind ' +
                                          str(sampKind) +
                                          'is not implemented.')
            header = struct.pack('>iihh', nSamples, sampPeriod, sampSize,
                                 sampKind)
            f.write(header)
            data.astype('>f').tofile(f)

import numpy as np


class Welford(object):
    """
    Implements Welford's algorithm for computing a running mean and
    standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html

    Can take a single value or iterables.

    Properties
    ----------
    mean     - returns the mean.
    std      - returns the std.
    meanfull - returns the mean and std of the mean.

    Examples
    --------
    >>> foo = Welford()
    >>> foo(range(100))
    >>> foo
    <Welford: 49.5 +- 29.0114919759>
    >>> foo([1] * 1000)
    >>> foo
    <Welford: 5.40909090909 +- 16.4437417146>
    >>> foo.mean
    5.409090909090906
    >>> foo.std
    16.44374171455467
    >>> foo.meanfull
    (5.409090909090906, 0.4957974674244838)
    """
    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def __call__(self, x):
        if hasattr(x, '__iter__'):
            self.consume(x)
        else:
            self.update(x)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        return 0 if self.k == 1 else np.sqrt(self.S / (self.k - 1))

    @property
    def meanfull(self):
        return self.mean, self.std / np.sqrt(self.k)

    def __repr__(self) -> str:
        return "<Welford: {} +- {}".format(self.mean, self.std)


if __name__ == '__main__':
    foo = Welford()
    foo(range(100))
    foo

    foo([1] * 1000)
    foo

    foo.mean
    foo.std
    foo.meanfull

import math
import numpy as np
import traceback


def psdfreqvec(npts=1024, fs=None, centerDC=False, freq_range='whole'):
    """
    Return a frequency vector in radians based on the number of points specified in NPTS.
    The vector returned assumes 2pi periodicity.

    Parameters
    ----------
    npts: specifies the number of sampling points.

    fs: specifies the sampling frequency in hertz. \
        By default fs is set to empty indicating normalized frequency.

    centerDC: specifies a boolean value in CENTERDC, \
        which indicates if zero hertz should be in the center of the frequency vector.
        CENTERDC can be one of these two values: False | True.

    freq_range: specifies the range of frequency in RANGE.
        RANGE can be one of the two strings: 'whole' | 'half'.

        Assuming CenterDC = false then:
            'whole' = [0, 2pi)
            'half'  = [0, pi] for even NPTS or [0, pi) for odd NPTS
        When CenterDC=true then:
            'whole' = (-pi, pi] for even NPTs or (-pi, pi) for odd NPTs
            'half'  = [-pi/2, pi/2] for even* NPTS or (-pi/2, pi/2) for odd NPTS
            * When NPTS is not divisible by 4, then the range is (-pi/2, pi/2).

        When freq_range='half' the frequency vector has length (NPTS/2+1) if NPTS is
        even**, and (NPTS+1)/2 if NPTS is odd***.

            ** If centerDC=True and the number of points specified is even is
            not divisible by 4, then the number of points returned is NPTS/2.
            This is to avoid frequency points outside the range [-pi/2 pi/2].

            *** If centerDC=True and the number of points NPTS specified is odd
            and (NPTS+1)/2 is even then the length of the frequency vector is
            (NPTS-1)/2.
    """
    # Compute the frequency grid.
    if fs is None:
        # Compute the whole frequency range, e.g., [0, 2pi) to avoid round off errors.
        fs = 2 * math.pi

    freq_res = fs / npts
    w = freq_res * np.linspace(0, npts - 1, npts)
    w = w[:, np.newaxis]

    # There can still be some minor round off errors in the frequency grid.
    # Fix the known points, i.e., those near pi and 2pi.
    Nyq = fs / 2
    half_res = freq_res / 2

    # Determine if npts is odd and calculate half and quarter of npts.
    is_npts_odd, half_npts, is_half_npts_odd, quarter_npts = _npts_info(npts)

    if is_npts_odd:
        # Adjust points on either side of Nyquist.
        w[half_npts] = Nyq - half_res
        w[half_npts + 1] = Nyq + half_res
    else:
        # Make sure we hit Nyquist exactly, i.e., pi or Fs/2.
        w[half_npts] = Nyq
    w[-1] = fs - freq_res

    # Get the right grid based on range, centerdc, etc.
    w = _final_grid(
        w, npts, Nyq, freq_range, centerDC, is_npts_odd, is_half_npts_odd, half_npts, quarter_npts
    )
    return np.reshape(w, -1)


def _npts_info(npts):
    """
    Determine if we're dealing with even or odd lengths of npts, 1/2 npts,
    and 1/4 npts.
    """
    # Determine if npts is odd.
    is_npts_odd = False if npts % 2 == 0 else True
    # Determine half the number of points.
    half_npts = npts // 2
    # Determine if half Npts is odd.
    is_half_npts_odd = False if half_npts % 2 == 0 else True
    # Determine a quarter of the number of points.
    quarter_npts = (half_npts + 1) // 2 if is_half_npts_odd else half_npts // 2

    return is_npts_odd, half_npts, is_half_npts_odd, quarter_npts


def _final_grid(w, npts, Nyq, freq_range, centerDC,
                is_npts_odd, is_half_npts_odd, half_npts, quarter_npts):
    """
    Calculate the correct grid based on user specified values for range, centerdc, etc.
    """
    if freq_range.lower() == 'whole':
        # Calculated by default.% [0, 2pi)
        if centerDC:    # (-pi, pi] even or (-pi, pi) odd
            neg_endpt = half_npts + 1 if is_npts_odd else half_npts
            w = np.concatenate((-np.flipud(w[1:neg_endpt]), w[:half_npts + 1]))
    elif freq_range.lower() == 'half':
        w = w[:half_npts + 1]   # [0, pi] even or [0, pi) odd

        # For even number of points that are not divisible by 4 you get
        # less one point to avoid going outside the [-pi/2 pi/2] range.
        if centerDC:    # [-pi/2, pi/2] even (-pi/2, pi/2) odd
            neg_endpt = quarter_npts if is_half_npts_odd else quarter_npts + 1
            w = np.concatenate((-np.flipud(w[1:neg_endpt]), w[:neg_endpt]))
            if npts % 4 == 0:
                # Make sure we hit pi/2 exactly when npts is divisible
                # by 4! In this case it's due to roundoff.
                w[-1] = Nyq / 2
    else:
        raise Exception('Unknown frequency range type!')
        traceback.print_exc()

    return w

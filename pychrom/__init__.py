from email.mime import base
import warnings
import numpy as np
import scipy.stats as stat
from numpy.linalg import norm
from numpy import linalg as LA
from sklearn import preprocessing
from scipy import sparse
from scipy.optimize import curve_fit
#to work on data
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse import linalg
from sklearn import preprocessing

class PyChromIsFusedException(Exception):
    pass

class PyChromPeakSearchException(Exception):
    pass

def smooth_data(y_data: np.ndarray,
                window: int = 5,
                polyorder: int = 0,
                deriv: int = 0,
                axis: int = -1): #use savgol filter to smooth data
    '''
    Function to smooth the data
    
    Parameters
    --------
    y_data : ndarray
    window : int, default = 5
    polyorder: int, default = 0
    deriv : int, default = 0
    
    Returns
    ---------
    smooth_data : ndarray
    '''
    smooth_data = savgol_filter(y_data, window, polyorder, deriv, axis=axis)
    return smooth_data

def baseline_arPLS(y: np.ndarray,
                    ratio: float = 1e-6,
                    lam: int = 10000,
                    niter: int = 10):
    '''
    Function to perform baseline correction using arPLS

    Parameters
    ---------
    y : ndarray
    ratio : float, default = 1e-6
        Governs the extent of asymmetry required of the fit.
        Larger values allow more negative-going regions.
        Smaller values disallow negative-going regions.
        P must be > 0 and < 1.\n
    lam : int, default = 10000
        Controls the amount of curvature allowed for the baseline.
        The smaller the lambda, the more curvature allowed in the fit baseline.\n
    niter : int, default = 10
        Number of iterations

    Returns
    --------
    d : np.ndarray
        Corrected baseline


    References
    ---------
    [1] Baseline correction with arPLS https://pubs.rsc.org/en/content/articlelanding/2015/AN/C4AN01061B#!divAbstract \n
    [2] https://stackoverflow.com/questions/29156532/python-baseline-correction-library/67509948#67509948
    '''
    L = len(y)
    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    crit = 1
    count = 0
    while crit > ratio:
        z = linalg.spsolve(W + H, W * y) # array to be subtracted from y
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
        crit = norm(w_new - w) / norm(w) # stop criterion
        w = w_new
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        count += 1 # number of iterations
        if count > niter:
            break        
    return d                                            # z = to be subtracted from y_data


def normalize(y_data : np.ndarray, axis_var = None) -> np.ndarray:
    '''
    Normalise 1D data between 0 to 1 based on max and min value

    Parameters
    --------
    y_data : ndarray

    Returns
    --------
    y_norm : ndarray

    '''
    # # Return Value Error
    # if type(y_data) != np.ndarray:
    #     raise ValueError(f'Wrong data type input. Expecting {np.ndarray}, got {type(y_data)}')
    if axis_var == None:
        shape = y_data.shape
        y_data_min_arr = np.amin(y_data, axis=axis_var)
        y_data_diff = (np.amax(y_data, axis=axis_var) - np.amin(y_data, axis=axis_var))
        y_norm = np.subtract(y_data, y_data_min_arr)
        y_data_diff = (np.amax(y_data, axis=axis_var) - np.amin(y_data, axis=axis_var))
        y_norm = np.true_divide(y_norm, y_data_diff)
    else:
        if axis_var == 1:
            shape = (-1,1)
        elif axis_var == 0:
            shape = (1,-1)
        else:
            raise ValueError('Wrong axis_var selection')
        y_data_min_arr = np.amin(y_data, axis=axis_var).reshape(shape)
        y_data_diff = (np.amax(y_data, axis=axis_var) - np.amin(y_data, axis=axis_var)).reshape(shape)
        y_norm = np.subtract(y_data, y_data_min_arr)
        y_data_diff = (np.amax(y_data, axis=axis_var) - np.amin(y_data, axis=axis_var)).reshape(shape)
        y_norm = np.true_divide(y_norm, y_data_diff)
    return y_norm

def peak_search(y_data: np.ndarray,
                prominence: float = 0.1) -> list[int]:
    '''
    Uses scipy find_peaks to search peaks across y_data array using prominence parameter

    Parameters
    --------
    y_data: np.ndarray
    prominence: float = 0.1

    Returns
    ---------
    peak_indices: list[int]

    '''
    y_data = normalize(y_data)
    peak_indices, _ = find_peaks(y_data, prominence)
    if len(peak_indices) >= 1:
        return list(peak_indices)
    else:
        warnings.warn('No peak found')
        return []

def peak_search_subset(initial_idx : int,
                        y_data : np.ndarray,
                        prominence : float = 0.1,
                        subset_range : int = 50) -> int:
    '''
    Search peak within an interval and returns the index of the most intense peak

    Parameters
    --------
    initial_idx : int
        Index where the search will be performed
    y_data : ndarray
        Input data where the peak will be searched
    prominence : float, default = 0.1
    subset_range : int
        Window to search for the peak in terms of index

    Returns
    --------
    highest_peak : int
        Index of the highest peak within the searched range
    '''
    # normalize data
    y_data = normalize(y_data)

    # range of peak lookup
    interval = subset_range
    lower_int = initial_idx - interval
    upper_int = initial_idx + interval

    # data trimming based on the interval
    y_trim = y_data[lower_int:upper_int]

    # lookup for the peaks
    peak_indices, _ = find_peaks(y_trim, prominence)
    peak_list = y_trim[peak_indices]

    if len(peak_indices) >= 1:
        # Search the index of the most intense peak
        highest_peak = np.where(y_trim == np.max(peak_list))

        # Define y-value of the highest peak
        y_peak = y_trim[highest_peak]

        # Return index from original dataset
        highest_peak = np.abs(y_data - y_peak).argmin()

        return highest_peak

    else:
        print('No peak found on subsearch. Returning initial index')
        return initial_idx

def peak_search_width(y_peak_idx: int,
                        y_data: np.ndarray,
                        rel_height: float = 0.99) -> tuple((int, int, float)):
    '''
    Find peak widths taking peak indexes and y-data

    Parameters
    ---------
    y_peak_idx : int
    y_data : ndarray
    rel_height : float, default = 0.99

    Returns
    ---------
    start_int : int
    end_int : int
    y_at_rel_height : float
        The y-value at the selected relative height
    '''
    # Transform into array and flatten into 1D
    y_peak_idx = np.array(y_peak_idx).flatten()

    # Find widths 
    width = peak_widths(y_data, peaks = y_peak_idx, rel_height=rel_height)

    # Define intervals
    start_int = width[2].astype(int).item()
    end_int = width[3].astype(int).item()

    # Define the y-value at the selected rel_height
    y_at_rel_height = width[1]

    return start_int, end_int, y_at_rel_height

def peak_integrate(y_data : np.ndarray,
                    peaks_heights_idx : int,
                    int_interval : tuple((int,int))=None,
                    rel_height : int=0.99) -> float:

    ''' Peak integration using the composite trapezoidal rule.

    Parameters
    ---------
    y_data : np.ndarray
    peaks_heights_idx : int
        Index of peak to be integrated

    Returns
    ---------
    integration : float
        Numerical value of the integration

    Notes
    ---------
    In this implementation, if no interval is provided, the function will find the range
    using the function `peak_search_width`.

    '''
    if int_interval == None:
        left_int, right_int, _ = peak_search_width(peaks_heights_idx, y_data, rel_height, return_idx=True)
        integration = np.trapz(y_data[left_int:right_int])
    else:
        integration = np.trapz(y_data[int_interval[0]:int_interval[-1]])

    return integration

def mean_center(y_data: np.ndarray, axis = 0) -> np.ndarray:
    '''
    Function to center data using np arrays

    Parameters
    ---------

    y_data : ndarray

    Returns
    --------

    y_centered : ndarray

    Notes
    --------
    '''
    # calculate the mean for each dimension (columns)
    y_centered = y_data - y_data.mean(axis=axis)

    return y_centered

def calculate_normalized_noise(y_data : np.ndarray) -> tuple((float, float)):
    '''
    Function to calculate instrumental noise

    Parameters
    --------
    y_data : ndarray
        The input raw data without any kind of smoothing

    Returns
    ---------
    norm_noise_std : float
        Calculated standard deviation between the y_i (raw data)
        and the y_mean (smoothed data)

    Notes
    ---------
    This implementation estimates the noise by calculating the standard deviation
    using the y_mean as the smoothed signal and y_i as the raw data.
    Firstly, data have its baseline corrected and it is normalized.
    Then Savitzky-Golay smoothing yields `y_mean`.
    Then, the residues are calculated by performing `y_mean - y_data`.
    From `y_mean - y_data` mean and standard deviation are calculated using:
    '''
    # Calculation of y_norm
    y_norm = normalize(y_data)

    # Calculate y_mean
    y_mean = smooth_data(y_norm, window=5, polyorder=0, deriv=0)

    y_noise = np.power(np.subtract(y_data, y_mean), 2) # (y_i - y_mean)^2
    noise_var = np.sum(y_noise)/len(y_noise.flatten())
    norm_noise_std = np.power(noise_var, 0.5)

    return norm_noise_std

def calculate_raw_noise(y_data : np.ndarray) -> tuple((float, float)):
    '''
    Function to calculate instrumental noise

    Parameters
    --------
    y_data : ndarray
        The input raw data without any kind of smoothing

    Returns
    ---------
    noise_std : float
        Calculated standard deviation between the y_i (raw data)
        and the y_mean (smoothed data)

    Notes
    ---------
    This implementation estimates the noise by calculating the standard deviation
    using the y_mean as the smoothed signal and y_i as the raw data.
    Savitzky-Golay smoothing yields `y_mean`.
    Then, the residues are calculated by performing `y_mean - y_data`.
    From `y_mean - y_data` mean and standard deviation are calculated using:
    '''

    y_mean = smooth_data(y_data, window=10, polyorder=0, deriv=0)
    y_noise = np.power(np.subtract(y_data, y_mean), 2) # (y_i - y_mean)^2
    noise_var = np.sum(y_noise)/len(y_noise.flatten())
    raw_noise_std = np.power(noise_var, 0.5)

    return raw_noise_std

def peak_purity(pda_rt_scan_idx : int, start_idx : int,
                end_idx : int, scans_chromatogram : np.ndarray,
                type_norm : str = "l2", similarity_thresh : int = 950) -> float:
    '''
    Function to calculate peak purity based on spectral similarity

    Parameters
    ---------
    pda_rt_scan_idx : int
        x data array containing the retention times
    start_idx, end_idx : int
        indexes representing the start and end of a peak
    scans_chromatogram : ndarray
    type_norm : str, default = "l2"
        type of normalisation used for scans
    similarity_thresh : int, default = 950
        minimum value to compute a scan as equal as the reference. Should range 900 and 1000

    Returns
    ---------
    perc_purity : float
        Percentage of how pure the peak is

    Notes
    --------
    The implementation is based on correlation coefficient between two spectra using dot product divided by norm
    
    [1] First implementation without smoothing and trimming shows poor similarity between neighbour spectra\n
    [2] Performing smoothing enhances a lot the similarity but the max they reach is 0.970\n
    [3] Agilent references says that concentration differences are expected, but the gradient can change the
    absorptivity. This can be corrected using reference spectra before and after peak\n
    Implementation steps
    1- Smooth all data\n
    2- Calculate correction matrix to mitigate effects of the gradient on absorption\n
    3- Normalise data\n
    4- Calculate similarity\n
    5- Count how many scans have similarity higher than 950\n
    6- Calculate this percentage\n
    7- Return this percentage as peak purity\n
    This implementation does not use mean centering as this increase the spectral difference between\n
    scans, therefore returning a wrong value.

    References
    ---------
    [1] https://www.chromatographyonline.com/view/peak-purity-liquid-chromatography-part-i-basic-concepts-commercial-software-and-limitations
    [2] https://www.agilent.com/cs/library/applications/5988-8647EN.pdf
    '''
    # Scan indexes of the peak
    scans_idx_arr = range((start_idx), (end_idx))

    # Data preprocessing
    scans_chromatogram = savgol_filter(scans_chromatogram, window_length=10, polyorder = 0, deriv=0, axis=1)

    '''Concentration correction'''
    # Get noise spectra indexes for correction - This may change due to peak tailing
    scan_before_idx = scans_idx_arr[0] - 5
    scan_after_idx = scans_idx_arr[-1] + 15

    # Get the noise scan
    scan_before = scans_chromatogram[scan_before_idx, :]
    scan_after = scans_chromatogram[scan_after_idx, :]

    # PDA scans size as tuple (row : num of scans, col : num of wavelengths)
    PDA_matrix_size = (len(scans_idx_arr), scans_chromatogram.shape[1])

    # Build empty correction matrix
    correction_matrix = np.zeros(PDA_matrix_size)

    # Calculate interpolation matrix to correct baseline
    # xp : x-points to be evaluated
    # fp : y at xp
    for wavelength in range(scans_chromatogram.shape[1]):
        xp = [scan_before_idx, scan_after_idx]
        fp = [scan_before[wavelength], scan_after[wavelength]]
        correction_matrix[:, wavelength] = np.interp(scans_idx_arr, xp, fp)

    '''Data correction'''
    # Slice the data for the region of interest
    scans_chromatogram_corrected = scans_chromatogram[start_idx:end_idx, :]

    # Data correction - eliminate the gradient effect on PDA
    scans_chromatogram_corrected = scans_chromatogram_corrected - correction_matrix

    '''Scans data transformation to calculate the similarity'''
    # Remove irrelevant region from scans
    scans_chromatogram_corrected = scans_chromatogram_corrected[:,20:]
    correction_matrix = correction_matrix[:,20:]
    
    # Normalise data across columns (axis = 1)
    scans_chromatogram_corrected = preprocessing.normalize(scans_chromatogram_corrected,
                                                            norm=type_norm, axis = 1)

    # Save apex scan after processing
    REF_PDA_intensities = scans_chromatogram_corrected[pda_rt_scan_idx - start_idx, :]

    '''Similarity calculations'''
    # Dot product between peak scans and apex scan
    dot_ref_spec = np.matmul(scans_chromatogram_corrected,
                                REF_PDA_intensities)

    # Norm product of the peak scans and apex scan
    norm_ref_spec = LA.norm(REF_PDA_intensities)*LA.norm(scans_chromatogram_corrected, axis=1)

    # Calculation of cos theta, i.e., correlation coefficient
    purity_scans = 1000*(np.true_divide(dot_ref_spec, norm_ref_spec))
    purity_scans = np.abs(purity_scans)

    # Computation of how many scans have similarity higher than 950
    peak_purity_percentage = ((purity_scans >= similarity_thresh).sum())/purity_scans.shape[0]

    return peak_purity_percentage

def is_fused_peak(peak_intervals : tuple((int, int)),
                    y_data : np.ndarray,
                    diff_smooth_window : int = 20) -> bool:
    '''Function to determine if a peak is fused, returning a bool value
    
    Parameters
    ---------
    peak_intervals : tuple(int, int)
    y_data : np.ndarray
    Returns
    ---------
    is_fused : bool
    Raises
    ---------
    ValueError
        If only peak is found. A strong indication of the absence of gaussian/lorentzian peak shape
    Notes
    ---------
    The implementation is based on the mathematical property of gaussian-shaped peaks. The second derivative
    is performed upon the y-data. When it is a perfect gaussian peak, there are two peaks and one valley. The
    two peaks represent the start and end of the peak, and the valley is the peak itself.
    In V1.00, some problems are rising from the way that the algorithm detects the multi peak. Some situations are resulting
    in only one peak and others, 5 peaks.
    For one peak situations, the problem was in the y_diff slice using the peak intervals. To solve it,
    increasing the the interval window was enough.

    References
    ---------
    [1] Empower Apex-track
    '''

    # Determine the 2nd derivative
    y_diff = np.diff(y_data, n=2)
    y_diff = smooth_data(y_diff, diff_smooth_window)

    # Slice the array within the interval provided, but a bit broader (+- 10)
    y_diff = y_diff[(peak_intervals[0]-10):(peak_intervals[1]+10)]

    # Fetch both peaks. Prominence must be > 0.5 since the baseline shifts to 0.5 after normalisation
    peaks_idx = peak_search(y_diff, prominence=0.6)
    is_fused = False
    if len(peaks_idx)==2:
        is_fused = False
    elif len(peaks_idx)>=3:
        is_fused = True
    else:
        error_msg = f'Unhandled condition at the is_fused_peaks. Length: {len(peaks_idx)}'
        raise PyChromIsFusedException(error_msg)

    return is_fused

def split_fused_peaks(peak_idx: int,
                        peak_intervals: tuple((int, int)),
                        y_data: np.ndarray,
                        diff_smooth_window: int = 20,
                        slice_window: int = 10) -> tuple((int, int)):
    '''Function to return the interval related to the peak of interest within the interval provided.

    Parameters
    ---------
    peak_idx : int
    peak_intervals : tuple of ints
    y_data : np.ndarray

    Returns
    ---------
    int_interval : tuple((int, int))
        New interval

    Notes
    ---------
    The implementation is based on the same function `is_fused_peak` using the 2nd derivative.
    The function will return the new interval based on the peaks detected in the 2nd deriv.
    It still has an issue with the peak intervals, because sometime it selects the starting point
    when the height is higher than 10%.
    For a next tweak, taking the height at 50% from the 2nd derivative peak may solve the problem.
    Implement a general solution for situations when more than 3 peaks are detected in the 2nd.

    References
    ---------
    '''
    
    # Determine the 2nd derivative
    y_diff = np.diff(y_data, n=2)
    y_diff = smooth_data(y_diff, diff_smooth_window)

    # Slice the array within the interval provided, but a bit broader
    y_diff_sliced = y_diff[(peak_intervals[0]-slice_window):(peak_intervals[1]+slice_window)]

    # Fetch both peaks. Prominence must be > 0.5 since the baseline shifts to 0.5 after normalisation
    peaks_list = peak_search(y_diff_sliced, prominence=0.6)
    peaks_list = [np.where(y_diff == y_diff_sliced[peak])[0] for peak in peaks_list]

    # Think in a better implementation for that

    '''Determine which `i` and `i+1` peak_idx is in, yielding
    int_start and int_end. From here, determine width for int_start'''
    # Iterates over the whole peak list
    for i in range(len(peaks_list)-1):
        # If the index is between an interval inside the peak list,
        # will execute interval selection
        if peaks_list[i] < peak_idx < peaks_list[i+1]:
            # If it is between the beginning
            int_interval = (peaks_list[i].item(), peaks_list[i+1].item())
            break

    return int_interval
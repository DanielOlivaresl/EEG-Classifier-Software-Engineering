import pygsp.graphs
import scipy.signal
import scipy.ndimage
import mne 
import numpy
import pygsp
# import pyemd
import pywt
import sklearn.decomposition
import numpy as np
from scipy import signal


def laplacianFilter(signal):
    laplacianSignal = scipy.ndimage.laplace(signal)
    return laplacianSignal

def calcularAnchoDeBanda(signal):

    freqSpectrum = numpy.fft.ftt(signal)
    modulationBand = scipy.calculateModulation(freqSpectrum)
    
    frequencyBand = scipy.calculateModulationBand(freqSpectrum)
    
    return modulationBand, frequencyBand


def spatialFeaturesFusion(signals):
    fusedSignals = scipy.spatialFusion(signals)
    return fusedSignals


def wvgsp(signals,conexions):
    graph = pygsp.Graph(conexions)
    
    return pygsp.signalGraph(graph,signals)
     

    
# def emdDecomposition(signal):
#     emd = pyemd.EMD()
    
#     imfs = emd(signal)
    
#     return imfs



def passBand(signal,lowCut,highCut,fs):
    #We normalize the frequencies by dividing by the Nyquist frequency 
    
    nyquist = 0.5 * fs
    low = lowCut / nyquist
    high = highCut / nyquist
    
    #We define the filter    
    sos = scipy.signal.butter(N=4,Wn=[lowCut,highCut],btype='bands')
    
    #Applying the filter to the signal    
    bandPassSignal = scipy.signal.sosfilt(sos,signal)
    return bandPassSignal


def applyHighPass(signal,highcut,fs):
    sos = scipy.signal.butter(4,highcut)
    
    highPassSignal = scipy.signal.sosfilt(sos,signal)
    
    return highPassSignal


def applyWavelet(signal,wavelet):
    coeffs = pywt.wavedec(signal,wavelet)
    return coeffs


def notchFilter(signal,freqs,Q = 30,fs=250):
    b,a = scipy.signal.iirnotch(freqs,Q,fs)
    
    
    notchsignal = scipy.signal.filtfilt(b,a,signal)
    
    return notchsignal


        
    
def ica(signals,ica):
    
    
    return ica.fit_transform(signals.T).T


#Methods for automatic artifact removal

def amplitudeThresh(icaSignal,thresh=100e-6):
    # Find the maximum absolute amplitude for each channel (component)
    max_amplitudes = np.max(np.abs(icaSignal), axis=1)

    # Identify components that exceed the amplitude threshold
    excluded_components = [i for i, amp in enumerate(max_amplitudes) if amp > thresh]

    # Set excluded components to zero
    cleaned_components = icaSignal.copy()
    for i in excluded_components:
        cleaned_components[i, :] = 0  # Set the component to zero along the time dimension

    return cleaned_components  # Return cleaned ICA components
    

def powerBasedRejection(icaSignal,thresh=0.5):
   
     # Calculate variance for each channel (component)
    variances = np.var(icaSignal, axis=1)

    # Identify components that exceed the variance threshold
    excluded_components = [i for i, var in enumerate(variances) if var > thresh]

    # Set excluded components to zero
    cleaned_components = icaSignal.copy()
    for i in excluded_components:
        cleaned_components[i, :] = 0  # Set the component to zero along the time dimension

    return cleaned_components  # Return cleaned ICA components

    





def rebuildSignal(components):
    reconstructed = numpy.sum(components,axis=1)
    return reconstructed


def filter_eeg(X: np.ndarray, sampling_freq: int = 250, notch_freq: float = 60.0, lowcut: float = 0.5,
               highcut: float = 30.0, scaling_factor: float = 50 / 1e6) -> np.ndarray:
    """
    Filter EEG data using notch and bandpass filters.

    Parameters:
    - X (np.ndarray): EEG data to be filtered with shape (trials, channels, samples).
    - sampling_freq (int, optional): Sampling frequency of the EEG data in Hz. Default is 250 Hz.
    - notch_freq (float, optional): Notch filter frequency in Hz (e.g., 60 Hz for powerline interference). Default is 60 Hz.
    - lowcut (float, optional): Low cutoff frequency for the bandpass filter in Hz. Default is 0.5 Hz.
    - highcut (float, optional): High cutoff frequency for the bandpass filter in Hz. Default is 30 Hz.
    - scaling_factor (float, optional): Scaling factor to convert the filtered data to µV (microvolts). Default is 50 µV / 1e6.

    Returns:
    - np.ndarray: Filtered EEG data with shape (trials, channels, samples) after applying notch and bandpass filters and scaling.
    """
    # Design the notch filter
    Q = 30  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=sampling_freq)

    # Design the bandpass filter
    nyquist_freq = 0.5 * sampling_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b_bandpass, a_bandpass = signal.butter(4, [low, high], btype='band')

    # Initialize filtered EEG data array
    filtered_eeg_data = np.zeros_like(X)

    # Apply the notch and bandpass filters to each trial and channel
    for trial in range(X.shape[0]):
        for channel in range(X.shape[1]):
            # Apply notch filter
            eeg_notch_filtered = signal.filtfilt(
                b_notch, a_notch, X[trial, channel, :])
            # Apply bandpass filter
            filtered_eeg_data[trial, channel, :] = signal.filtfilt(
                b_bandpass, a_bandpass, eeg_notch_filtered)

    # Scale the filtered EEG data to ±50 µV
    filtered_eeg_data *= scaling_factor

    return filtered_eeg_data
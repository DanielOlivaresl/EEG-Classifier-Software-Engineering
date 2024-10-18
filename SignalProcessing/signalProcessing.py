import pygsp.graphs
import scipy.signal
import scipy.ndimage
import mne 
import numpy
import pygsp
import pyemd
import pywt
import sklearn.decomposition



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
     

    
def emdDecomposition(signal):
    emd = pyemd.EMD()
    
    imfs = emd(signal)
    
    return imfs



def passBand(signal,lowCut,highCut,fs):
    #We normalize the frequencies by dividing by the Nyquist frequency 
    
    nyquist = 0.5 * fs
    low = lowCut / nyquist
    high = highCut / nyquist
    
    #We define the filter    
    sos = scipy.signal.butter(N=4,Wn=[lowCut,highCut],btype='bandpass',output='sos',fs=fs)
    
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


def notchFilter(signal,freqs,Q,fs):
    b,a = scipy.signal.iirnotch(freqs,Q,fs)
    
    
    notchsignal = scipy.signal.filtfilt(b,a,signal)
    
    return notchsignal


        
    
def ica(signals):
    ica = sklearn.decomposition.FastICA(n_components=len(signals))
    
    separatedSignals = ica.fit_transform(signals)
    return separatedSignals



def rebuildSignal(components):
    reconstructed = numpy.sum(components,axis=1)
    return reconstructed


    
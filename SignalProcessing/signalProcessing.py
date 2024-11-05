import pygsp.graphs
import scipy.signal
import scipy.ndimage
import mne 
import numpy
import pygsp
# import pyemd
import pywt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA


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


        
    
def apply_ICA(X: np.ndarray, n_components: Optional[int] = 2) -> np.ndarray:
    """
    Apply ICA to each trial in the data and flatten the output to create feature vectors.
    Fits ICA on the entire dataset and applies the transformation.
    
    Parameters:
    - X: NumPy array of shape (trials, channels, time_steps)
    - n_components: Number of ICA components to extract. If None, all components are used.
    
    Returns:
    - ica_df: Transformed and flattened, where each row is a flattened feature vector for an trial.
    """
    
    # Prepare the data for ICA: shape should be (n_samples, n_features)
    num_trials, num_channels, num_time_steps = X.shape
    
    # Reshape and apply ICA on the entire dataset
    X_reshaped = X.reshape(num_trials, num_channels * num_time_steps)
    ica = FastICA(n_components=n_components, random_state=0)
    ica_data = ica.fit_transform(X_reshaped)  # Fit and transform on the entire dataset
    
    return ica_data

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





def convert_eeg_vmrk_to_csv(vhdr_file: str, output_csv_file: str):
    """
    Convert BrainVision EEG data and event markers to CSV format with remapped column names.

    Parameters:
    vhdr_file (str): Path to the .vhdr file.
    output_csv_file (str): Path to the output CSV file.
    """
    if not vhdr_file.endswith('.vhdr'):
        raise ValueError("The input file must be a .vhdr file.")
    
    if not os.path.exists(vhdr_file):
        raise FileNotFoundError(f".vhdr file not found: {vhdr_file}")
    
    # Load the EEG data using the .vhdr file
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    
    # Get the data as a NumPy array
    data, times = raw.get_data(return_times=True)
    
    # Create a DataFrame for EEG data
    df_eeg = pd.DataFrame(data.T, columns=raw.ch_names)
    df_eeg['Time'] = times

    # Try to load events from the .vmrk file referenced by the .vhdr file
    try:
        events, event_ids = mne.events_from_annotations(raw)
        df_events = pd.DataFrame(events, columns=['Sample', 'Previous', 'EventID'])
        df_events['Time'] = df_events['Sample'] / raw.info['sfreq']
        df_events['State'] = df_events['EventID']
        df_events.drop(columns=['Sample', 'Previous', 'EventID'], inplace=True)  # Keep only time and state
    except ValueError as e:
        print(f"Warning: Could not extract annotations from {vhdr_file}: {str(e)}")
        df_events = pd.DataFrame(columns=['Time', 'State'])
    
    # Merge EEG data and events DataFrames on time using the nearest match
    df = pd.merge_asof(df_eeg, df_events, on='Time', direction='nearest')
    
    # Rename EEG columns to have consistent 'EEG {i}' naming
    eeg_columns = [f'EEG {i+1}' for i in range(len(raw.ch_names))]
    df.rename(columns=dict(zip(raw.ch_names, eeg_columns)), inplace=True)
    
    # Include only the necessary columns: EEG channels, Time, and State
    columns_to_include = eeg_columns + ['Time', 'State']
    df_filtered = df[columns_to_include]
    
    # Save the final DataFrame to a CSV file
    df_filtered.to_csv(output_csv_file, index=False)
    print(f"Data from {vhdr_file} has been converted and saved to {output_csv_file}")



def process_eeg(TIME_STEPS:int = 1200, included_states: List[str] =["Up", "Down", "Left", "Right", "Select"], subject_folder :str ='./NPYData/Subject_0')->Tuple[np.ndarray, np.ndarray]:
    """
        Process EEG data files from a specified subject folder and extract relevant EEG data segments.

        Parameters:
        TIME_STEPS (int): The number of time steps for each EEG data segment. Default is 1200.
        included_states (list): List of states to include in the processing. Default is ["Up", "Down", "Left", "Right", "Select"].
        subject_folder (str): Path to the folder containing EEG data files for the subject. Default is './NPYData/Subject_0'.

        Returns:
        tuple: A tuple containing:
            - X (np.ndarray): A NumPy array of shape (number of samples, number of EEG channels, TIME_STEPS) containing the processed EEG data.
            - Y (np.ndarray): A NumPy array of shape (number of samples,) containing the corresponding states for each EEG data segment.

        Notes:
        - The function reads CSV files from the specified subject folder.
        - Each CSV file is expected to contain columns 'EEG 1' to 'EEG 8' for EEG channels and 'State' for the state labels.
        - The function groups the EEG data by state transitions and extracts segments of length TIME_STEPS.
        - If a segment is shorter than TIME_STEPS, it is padded with zeros.
        - The function ensures that all extracted segments have the same shape.
        - If there are inconsistent shapes, the function filters out those segments and only retains the consistent ones.
    """

    files = os.listdir(subject_folder)
    subjectID = subject_folder.split('_')[-1]
    dfs = []
    # Read and process each file
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(subject_folder, file))
        elif file.endswith('.npy'):
            df = pd.DataFrame(np.load(os.path.join(subject_folder, file),allow_pickle=True)).rename({0:'EEG 1', 1:'EEG 2', 2:'EEG 3', 3:'EEG 4', 4:'EEG 5', 5:'EEG 6', 6:'EEG 7', 7:'EEG 8', 17:'State'}, axis=1)
        else:
            #Skipping non-CSV file
            continue
        df['Subject'] = subjectID
        dfs.append(df[['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8', 'State', 'Subject']])

    # Process EEG data for each state
    all_state_data = []

    for df in dfs:
        state_groups = df.groupby((df['State'] != df['State'].shift()).cumsum())

        for _, data in state_groups:
            state = data['State'].iloc[0]
            if state in included_states:
                eeg_data = np.transpose(data[['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6', 'EEG 7', 'EEG 8']].values)[:,:TIME_STEPS]
                # apply padding if timesteps are smaller than 1200
                if eeg_data.shape[1] < TIME_STEPS:
                    pad_width = TIME_STEPS - eeg_data.shape[1]
                    eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                else:
                    eeg_data = eeg_data[:, :TIME_STEPS]

                all_state_data.append(pd.DataFrame({'State': [state], 'EEG Data': [eeg_data]}))

    # Concatenate the processed data
    final_df = pd.concat(all_state_data, ignore_index=True)

    # Fetch the list of arrays
    data_list = final_df['EEG Data'].values
    state_list = final_df['State'].values

    # Check the shapes of all arrays
    shapes = [arr.shape for arr in data_list]

    # Ensure all shapes are the same
    if len(set(shapes)) == 1:
        # All arrays have the same shape, so convert the list to a NumPy array
        X = np.array([item for item in data_list])
        Y = np.array([state for state in final_df['State'].values])
    else:
        # Print the shapes that are inconsistent
        print("Inconsistent shapes found:", set(shapes))
        # Filter and store only consistent shapes of data
        X = np.array([item for item in data_list if item.shape[1] == TIME_STEPS])
        Y = np.array([state for item, state in zip(data_list, state_list) if item.shape[1] == TIME_STEPS])

    X = X.astype('float64')

    return X, Y
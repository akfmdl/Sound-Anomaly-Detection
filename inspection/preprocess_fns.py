import cv2 as cv
import os
import numpy as np
from nptdms import TdmsFile as td
import librosa

class Preprocess():
    def extract_mel_features(self, file, params):
        signal = self.load_tdms_file(file)
        # Compute a mel-scaled spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=params["sampling_rate"],
            n_fft=params["n_fft"],
            hop_length=params["n_step"],
            n_mels=params["n_mels"]
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return self.convert(log_mel_spectrogram, {"target_type_min": 0, "target_type_max": 255, "target_type": np.uint8})

    def extract_signal_features(self, file, params):
        signal = self.load_tdms_file(file)
        # Compute a mel-scaled spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=params["sampling_rate"],
            n_fft=params["n_fft"],
            hop_length=params["n_step"],
            n_mels=params["n_mels"]
        )

        # Convert to decibel (log scale for amplitude):
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Generate an array of vectors as features for the current signal:
        features_vector_size = log_mel_spectrogram.shape[1] - (params["stride"]*params["frames"]) + params["stride"]
        
        # Skips short signals:
        dims = params["frames"] * params["n_mels"]
        if features_vector_size < 1:
            return np.empty((0, dims), np.float32)
        
        # Build N sliding windows (=frames) and concatenate them to build a feature vector:
        features = np.zeros((features_vector_size, dims), np.float32)
        for t in range(params["frames"]):
            features[:, params["n_mels"]*t:params["n_mels"]*(t+1)] = log_mel_spectrogram[:, params["stride"]*t:(params["stride"]*t)+features_vector_size].T
            
        return self.convert(features, {"target_type_min": 0, "target_type_max": 255, "target_type": np.uint8})
    
    def load_tdms_file(self, tdms_file):
        tdms_data = td.read(tdms_file)
        # Read tdms_data with Pandas (other options: absolute_time=True, scaled_data=False)
        tdms_df = tdms_data.as_dataframe(time_index=True)
        try:
            # return tdms_df.values[:,0][:119040]
            return tdms_df.values[:,0][:102400] # remove noise
        except:
            return None

    def convert(self, img, params):
        imin = img.min()
        imax = img.max()

        a = (params["target_type_max"] - params["target_type_min"]) / (imax - imin)
        b = params["target_type_max"] - a * imax
        new_img = (a * img + b).astype(params["target_type"])
        return new_img

    def gray_to_rgb(self, gray_img):
        return cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
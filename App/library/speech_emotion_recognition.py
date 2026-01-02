## Basics ##
import time
import os
import numpy as np

## Audio Preprocessing ##
import pyaudio
import wave
import librosa
from scipy.stats import zscore

## Time Distributed CNN ##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM

class speechEmotionRecognition:

    def __init__(self, subdir_model=None):
        # Load prediction model
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)

        # Emotion encoding
        self._emotion = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Neutral', 5: 'Sad', 6: 'Surprise'
        }

    def voice_recording(self, filename, duration=5, sample_rate=16000, chunk=1024, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []
        print('* Start Recording *')
        stream.start_stream()
        start_time = time.time()
        current_time = time.time()

        while (current_time - start_time) < duration:
            data = stream.read(chunk)
            frames.append(data)
            current_time = time.time()

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('* End Recording *')

        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    	# Replace NaNs and Infs with zeros
    	y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    	# Compute the Short-Time Fourier Transform (STFT)
    	stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
    	spectrogram = np.abs(stft) ** 2  # Power spectrogram

    	# Compute the mel spectrogram from the power spectrogram
    	mel_spect = librosa.feature.melspectrogram(S=spectrogram, sr=sr, n_mels=n_mels, fmax=fmax)

    	# Convert to log scale (dB)
    	mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    	return mel_spect_db


    def frame(self, y, win_step=64, win_size=128):
        y = np.asarray(y)

        if y.ndim == 3:
            batch_size, n_features, time_steps = y.shape
            if time_steps < win_size:
                raise ValueError(f"Too short for framing: time_steps={time_steps}, win_size={win_size}")
            nb_frames = 1 + int((time_steps - win_size) / win_step)
            frames = np.zeros((batch_size, nb_frames, n_features, win_size), dtype=np.float16)
            for t in range(nb_frames):
                frames[:, t, :, :] = y[:, :, t * win_step:t * win_step + win_size]
            return frames

        elif y.ndim == 2:
            y = y[np.newaxis, :, :]
            return self.frame(y, win_step=win_step, win_size=win_size)

        else:
            raise ValueError(f"Input shape not supported: y.shape={y.shape}")

    def build_model(self):
        K.clear_session()
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        y = TimeDistributed(Conv2D(64, (3, 3), padding='same'), name='Conv_1')(input_y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(y)
        y = TimeDistributed(Dropout(0.2))(y)

        y = TimeDistributed(Conv2D(64, (3, 3), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D((4, 4), padding='same'))(y)
        y = TimeDistributed(Dropout(0.2))(y)

        y = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D((4, 4), padding='same'))(y)
        y = TimeDistributed(Dropout(0.2))(y)

        y = TimeDistributed(Conv2D(128, (3, 3), padding='same'))(y)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('elu'))(y)
        y = TimeDistributed(MaxPooling2D((4, 4), padding='same'))(y)
        y = TimeDistributed(Dropout(0.2))(y)

        y = TimeDistributed(Flatten())(y)
        y = LSTM(256, return_sequences=False, dropout=0.2)(y)
        y = Dense(7, activation='softmax')(y)

        model = Model(inputs=input_y, outputs=y)
        return model

    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):
        y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)

        # Pad if too short
        if y.shape[0] < chunk_size:
            y = np.pad(y, (0, chunk_size - y.shape[0]), mode='constant')

        chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
        chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])
        y = np.asarray(list(map(zscore, chunks)))
        mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))
        mel_spect_ts = self.frame(mel_spect)
        X = mel_spect_ts.reshape(mel_spect_ts.shape[0], mel_spect_ts.shape[1], mel_spect_ts.shape[2], mel_spect_ts.shape[3], 1)

        if predict_proba:
            predict = self._model.predict(X)
        else:
            predict = np.argmax(self._model.predict(X), axis=1)
            predict = [self._emotion.get(emotion) for emotion in predict]

        K.clear_session()
        timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
        timestamp = np.round(timestamp / sample_rate)

        return [predict, timestamp]

    def prediction_to_csv(self, predictions, filename, mode='w'):
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS\n")
            for emotion in predictions:
                f.write(str(emotion) + '\n')

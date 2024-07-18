import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from utils.constants import LENGHT_KEYPOINTS

from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
from keras.regularizers import l2 # type: ignore


def get_model(max_legth_frames, output_lenght: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_legth_frames, LENGHT_KEYPOINTS), kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_lenght, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
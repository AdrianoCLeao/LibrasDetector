import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from utils.utils import get_word_ids, get_sequences_and_labels
from utils.constants import *

import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.utils import to_categorical # type: ignore

def training_model(model_path, model_num:int, epochs=NUM_EPOCHS):
    word_ids = get_word_ids(KEYPOINTS_PATH)
    
    sequences, labels = get_sequences_and_labels(word_ids, model_num)

    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    model = get_model(int(model_num), len(word_ids))
    model.fit(X, y, epochs=epochs)
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    model_num = 18
    model_path = os.path.join(MODELS_FOLDER_PATH, f'actions_{model_num}.keras')
    training_model(model_path, model_num)


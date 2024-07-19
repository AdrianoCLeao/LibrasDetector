import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from utils.utils import *
from utils.constants import *

def create_keypoints(word_id, words_path, hdf_path):
    data = pd.DataFrame([])
    frames_path = os.path.join(words_path, word_id)

    with Holistic() as model_holistic:
        print(f'Criando keypoints de "{word_id}"...')
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)
        n_sample = 1

        for sample_name in sample_list:
            sample_path = os.path.join(frames_path, sample_name)
            frames_count = len(os.listdir(sample_path))

            model_num = "7" if MIN_LENGHT_FRAMES <= frames_count <= 7 else "12" if frames_count <= 12 else "18"

            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, model_num, keypoints_sequence)

            print(f"{n_sample}/{sample_count}", end="\r")
            n_sample+=1
    
    data.to_hdf(hdf_path, key="data", mode="w")
    print(f"Keypoints created! ({n_sample}) amostras")

if __name__ == "__main__":
    create_folder(KEYPOINTS_PATH)

    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]

    for word_id in word_ids:
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        create_keypoints(word_id, FRAME_ACTIONS_PATH, hdf_path)


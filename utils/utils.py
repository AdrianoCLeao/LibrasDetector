import os
import cv2
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
import numpy as np
import pandas as pd
from typing import NamedTuple
from utils.constants import *

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def there_hand(results: NamedTuple) -> bool:
    return results.left_hand_landmarks or results.right_hand_landmarks

def get_word_ids(path):
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)
        if ext == ".h5":
            out.append(name)
    return out

def draw_keypoints(image, results):
    draw_landmarks(
        image,
        results.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
    )
    draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        DrawingSpec(color=(22,44,121), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
    )
    draw_landmarks(
        image,
        results.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
    )
    
def save_frames(frames, output_folder):
    for num_frame, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{num_frame + 1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_keypoints(model, path):
    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        results = mediapipe_detection(frame, model)
        kp_frame = extract_keypoints(results)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

def insert_keypoints_sequence(df, n_sample:int, model_num:str, kp_seq):
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'model_num': model_num, 'frame':frame+1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])

    return df

def get_sequences_and_labels(words_id, model_num:int):
    sequences, labels = [], []

    for word_index, word_id in enumerate(words_id):
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        data = pd.read_hdf(hdf_path, key='data')
        for num, df_by_model_num in data.groupby('model_num'):
            if model_num == int(num):
                for _, df_sample in df_by_model_num.groupby('sample'):
                    seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
                    sequences.append(seq_keypoints)
                    labels.append(word_index)
    
    return sequences, labels



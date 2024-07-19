import os
import cv2

FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5,30)

MIN_LENGHT_FRAMES = 5
LENGHT_KEYPOINTS = 1662
MODEL_NUMS = [18,7,7]

ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODELS_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODELS_PATH = [os.path.join(MODELS_FOLDER_PATH, f"actions_{model_num}.keras") for model_num in MODEL_NUMS]
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")

NUM_EPOCHS = 5

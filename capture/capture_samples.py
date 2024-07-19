import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from utils.utils import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from utils.constants import FONT, FONT_POS, FONT_SIZE, ROOT_PATH, FRAME_ACTIONS_PATH
from datetime import datetime

def capture_sample(path, margin_frame=2, min_cant_frames=5):
    create_folder(path)

    count_frame = 0
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        while video.isOpened():
            ret, frame = video.read()
            image = frame.copy()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)

            if there_hand(results):
                count_frame+=1
                if count_frame > margin_frame:
                    cv2.putText(image, "Capturando.", FONT_POS, FONT, FONT_SIZE, (255,50,0))
                    frames.append(np.asarray(frame))

            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-margin_frame]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f'sample_{today}')
                    create_folder(output_folder)
                    save_frames(frames, output_folder)

                frames = []
                count_frame = 0
                cv2.putText(image, "Pronto para gravar.", FONT_POS, FONT, FONT_SIZE, (0,220,100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Coletando amostras em"{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "Maquina"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_sample(word_path)


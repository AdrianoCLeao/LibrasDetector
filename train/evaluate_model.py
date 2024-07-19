import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from utils.utils import *
from utils.constants import *
from utils.text_to_speech import text_to_speech

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model # type: ignore

def evaluate_model(src=None, threshold=0.5):
    count_frame = 0
    kp_sequence, sentence = [], []
    word_ids = get_word_ids(KEYPOINTS_PATH)
    model = load_model(MODELS_PATH[0])  # Carregando apenas o modelo 18

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)

        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)

            if there_hand(results):
                kp_sequence.append(extract_keypoints(results))
                count_frame += 1

            elif count_frame >= MIN_LENGHT_FRAMES:
                kp_sequence = pad_secuences(kp_sequence, 18)
                res = model.predict(np.expand_dims(kp_sequence, axis=0))[0]

                if res[np.argmax(res)] > threshold:
                    print(res[np.argmax(res)])
                    sent = word_ids[np.argmax(res)].split('-')[0]
                    sentence.insert(0, sent)
                    text_to_speech(sent)

                count_frame = 0
                kp_sequence = []

            if not src:
                cv2.rectangle(frame, (0,0), (640,35), (2445, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))

                draw_keypoints(frame, results)
                cv2.imshow('Tradutor', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()

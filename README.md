# Brazilian Handsign Detection

<img src="https://github.com/user-attachments/assets/36c6c711-d5f1-48e0-ae6f-c5ddf566440c" alt="Alt text" width="450">

This project aims to create a machine learning model that can identify, translate, and speak from live video of Libras (Brazilian Sign Language). It uses Google's Mediapipe to detect key points and Keras from TensorFlow to create the model.

## Project Structure

- `main.py`: Main file that initializes the GUI application for video recording and translation.
- `utils/`: Contains utility functions for data processing and keypoint detection.
  - `utils/utils.py`
  - `utils/constants.py`
  - `utils/text_to_speech.py`
- `models/`: Pre-trained models for gesture translation.
- `train/`: Scripts for model training.
  - `training_model.py`
  - `evaluate_model.py`
- `capture/`: Scripts for capturing and processing new data.
  - `capture_samples.py`
  - `create_keypoints.py`

## How to Use

1. **Create a Virtual Environment**

To isolate the project's dependencies, it is recommended to create a virtual environment using `venv`.

```bash
python -m venv venv
```
2. Activate the Virtual Environment
Activate the created virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS and Linux:
```bash
source venv/bin/activate
```
3. Install the Requirements
With the virtual environment activated, install the project's requirements:
```bash
pip install -r requirements.txt
```
4. Run the Application
Execute the main.py file to start the GUI application:
```bash
python main.py
```
## Contributions
Feel free to contribute to this project. You can open issues and pull requests to discuss and implement improvements.

## License
This project is under the MIT license. Check LICENSE.md for more details.

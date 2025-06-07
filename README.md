# License-Plate-detecton

This repository using `YOLOv8` to detect Taiwan License-Plate and use `openOCR.text_recognizer` to recognize Taiwan License


## Install the required Python libraries:

```sh
pip install -r requirements.txt
```

## Usage

### Inference one image/video
```sh
cd YOUR_REPO_DIR
python inference.py --file_path YOUR_FILE_PATH
```

### Config

- `weights_path`: Path to the weights file (default: models/best.pt)
- `file_path`: Path to the media file (image or video)
- `image_output_path`: Path to the image output (default: ./medias/output.jpg)
- `video_output_path`: Path to the video output (default: ./medias/output.mp4)
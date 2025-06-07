# License-Plate-detecton

This repository using `YOLOv8` to detect Taiwan License-Plate and use `openOCR.text_recognizer` to recognize Taiwan License

[Demo Video](https://youtu.be/U4SrPdsBx_Q?si=ucI-FYp7G-viJuHk)

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

## Acknowledgement

A pretrained weight and code are borrowed from
- [YOLOv8-License-Plate-Insights](https://github.com/yihong1120/YOLOv8-License-Plate-Insights/tree/main)

Thanks for their excellent work!
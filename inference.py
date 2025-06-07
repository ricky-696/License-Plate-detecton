import argparse
import cv2
import numpy as np

from openocr import OpenOCR
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class CarLicensePlateDetector:
    """
    A class to detect and recognize license plates on cars using the YOLO model and OCR.
    """

    def __init__(self, weights_path: str):
        """
        Initializes the CarLicensePlateDetector with the given weights.

        Args:
            weights_path (str): The path to the weights file for the YOLO model.
        """
        self.model = YOLO(weights_path)
        self.openOCR = OpenOCR(backend='onnx', device='cpu')

    def recognize_license_plate(self, img: np.ndarray) -> np.ndarray:
        """
        Recognizes the license plate in an image and draws a rectangle around it.

        Args:
            img (np.ndarray)

        Returns:
            np.ndarray: The image with the license plate region marked and annotated with the recognized text.
        """
        annotator = Annotator(img, line_width=5, pil=True)
        results = self.model.predict(img, save=False)
        boxes = results[0].boxes.xyxy

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Extract license plate text from the ROI
            roi = img[y1:y2, x1:x2]
            license_plate = self.extract_license_plate_text(roi)

            print(f"License: {license_plate}")
            annotator.box_label(box=[x1, y1, x2, y2], label=license_plate)

        return annotator.result()

    def load_image(self, img_path: str) -> np.ndarray:
        """
        Loads an image from the specified path.

        Args:
            img_path (str): The path to the image file.

        Returns:
            np.ndarray: The loaded image.
        """
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1].copy()  # Convert BGR to RGB

    def extract_license_plate_text(self, roi: np.ndarray) -> str:
        """
        Extracts text from a given image region (Region of Interest, ROI) using OCR.

        Args:
            roi (np.ndarray): The region of the image (as a NumPy array) that potentially contains a license plate.

        Returns:
            str: The extracted license plate text, or an empty string if extraction fails.
        """
        results = self.openOCR.text_recognizer(img_numpy_list=[roi], batch_num=6)
        
        try:
            plate_text = results[0]['text']
        except (IndexError, KeyError, TypeError):
            plate_text = ""
        
        return plate_text

    def process_video(self, video_path: str, output_path: str) -> None:
        """
        Processes a video file to detect and recognize license plates in each frame.

        Args:
            video_path (str): The path to the video file.
            output_path (str): The path where the output video will be saved.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0,
                             (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                annotated_frame = self.recognize_license_plate(frame)
                out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            else:
                break

        # Release everything when done
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="License Plate Detection")
    parser.add_argument('--weights_path', type=str, default='models/best.pt', help='Path to the weights file (e.g., models/best.pt)')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the media file (image or video)')
    parser.add_argument('--image_output_path', type=str, default='./medias/output.jpg', help='Path to the image output')
    parser.add_argument('--video_output_path', type=str, default='./medias/output.mp4', help='Path to the video output')

    args = parser.parse_args()

    detector = CarLicensePlateDetector(args.weights_path)

    if args.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(args.file_path)
        img = detector.load_image(args.file_path)
        recognized_img = detector.recognize_license_plate(img)
        
        cv2.imwrite(args.image_output_path, cv2.cvtColor(recognized_img, cv2.COLOR_RGB2BGR))
        print(f"Saved the image with the license plate to {args.image_output_path}")
    elif args.file_path.lower().endswith(('.mp4', '.mov', '.avi')):
        detector.process_video(args.file_path, args.video_output_path)
        print(f"Saved the processed video to {args.video_output_path}")
    else:
        print("Unsupported media format")


if __name__ == '__main__':
    main()

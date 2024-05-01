from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import sys
import torch

print(torch.cuda.is_available())  # Check if CUDA is available

# Get all license plates in image
def get_plates(result, img):
    images = []  # Store all license plates
    boxes = result[0].boxes  # List of all coordinates of license plates

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get license plate coordinates
        images.append(img[y1:y2, x1:x2].copy())  # Crop license plate image

    return images

# OCR
def get_LP_number(ocr, image):
    result = ocr.ocr(image, det=False)
    plate_number = ""

    for line in result:
        for word in line:
            plate_number += word[1][0]  # Concatenate characters

    return plate_number

# Process single image
# Draw rectangle around plates and LP number
def draw_box(result, img):
    boxes = result[0].boxes  # All coordinates of plates
    plate_numbers = [get_LP_number(ocr, img[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]) for box in boxes]  # All predicted LP numbers

    # For each LP coordinates and each predicted LP number of that LP
    for box, pnum in zip(boxes, plate_numbers):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get license plate coordinates

        # Draw rectangle around the LP
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fill background of the predicted LP number
        cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
        text_xy = (x1 + 2, y1 - 5)  # Coordinate of predicted LP number

        # Add predicted LP number
        cv2.putText(img, pnum, text_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img

# Process video
def video_draw_box(vid_path, model, ocr):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20, (width, height))

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        result = model(frame)  # Predict position of LPs
        frame = draw_box(result, frame)  # Draw rectangle and predicted LP number for current frame
        out.write(frame)  # Write to output.mp4
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Get weights and file_dir
        pre_trained_model = "runs/detect/train/weights/best.pt"
        media_type = sys.argv[1]
        file_dir = sys.argv[2]

        # Create model and OCR
        model = YOLO(pre_trained_model)
        ocr = PaddleOCR(lang='en')  # Create OCR instance

        if media_type == "-image":
            img = cv2.imread(file_dir)
            result = model(img)
            img = draw_box(result, img)
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            cv2.imwrite("predicted-" + file_dir, img)

        elif media_type == "-video":
            video_draw_box(file_dir, model, ocr)

    else:
        print("Usage: python script.py <-image/-video> <file_path>")
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os
from io import BytesIO
from PIL import Image
import streamlit as st

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_unique_color(class_index):
    # Generate a unique color based on the class index
    np.random.seed(class_index)
    color = list(np.random.random(size=3))
    return tuple(color)

def detect_objects_image(image_path, confidence_threshold=0.5, save_path="output_image.png"):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

    # Load image
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB

    # Inference
    results = model(img)

    # Filter results based on confidence threshold
    pred = results.xyxy[0]
    pred = pred[pred[:, 4] > confidence_threshold]

    # Draw rectangles and save image
    if len(pred) > 0:
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for det in pred:
            box = det[:4].cpu().numpy()
            confidence = det[4].cpu().numpy()
            class_index = int(det[5].cpu().numpy())  # Class index

            # Generate a unique color for each class
            color = get_unique_color(class_index)

            # Normalize color values to be in the range of 0 to 1
            color_normalized = Normalize()(color)

            # Draw rectangle
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor=color_normalized, facecolor='none')
            ax.add_patch(rect)

            # Add object name and confidence text
            class_name = model.names[class_index]
            plt.text(box[0], box[1], f'{class_name}: {confidence:.2f}', color=color_normalized)

        # Save the figure
        plt.savefig(save_path)
        plt.close()
        print(f"Object detection results saved to '{save_path}'")
    else:
        print("No objects detected.")

def detect_objects_video(video_path, confidence_threshold=0.5, save_path="output_video.mp4"):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

    cap = cv2.VideoCapture(video_path)
    frames_data = []

    # Get video properties
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = cap.get(5)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Inference
        results = model(frame)

        # Filter results based on confidence threshold
        pred = results.xyxy[0]
        pred = pred[pred[:, 4] > confidence_threshold]

        # Draw rectangles on the frame
        for det in pred:
            box = det[:4].cpu().numpy()
            confidence = det[4].cpu().numpy()
            class_index = int(det[5].cpu().numpy())  # Class index

            # Generate a unique color for each class
            color = get_unique_color(class_index)

            # Draw rectangle
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)

            # Add object name and confidence text
            class_name = model.names[class_index]
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save the frame
        out.write(frame)

    cap.release()
    out.release()
    print(f"Object detection results saved to '{save_path}'")

def main():
    st.title("Object Detection with YOLOv5")
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        file_path = "temp_file"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.type == "image/jpeg":
            detect_objects_image(file_path)
            st.image("output_image.png", caption="Processed Image", use_column_width=True)

        elif uploaded_file.type == "video/mp4":
            detect_objects_video(file_path)
            st.video("output_video.mp4")

if __name__ == "__main__":
    main()
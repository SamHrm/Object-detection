from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Specify the upload folder and allowed extensions for images
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Define available object classes
available_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_unique_color(class_index):
    # Generate a unique color based on the class index
    np.random.seed(class_index)
    color = list(np.random.random(size=3))
    return tuple(color)

def detect_objects(image_path, object_class, confidence_threshold=0.5):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

    # Load image
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB

    # Inference
    results = model(img)

    # Filter results based on confidence threshold and selected object class
    pred = results.xyxy[0]
    pred = pred[(pred[:, 4] > confidence_threshold) & (pred[:, 5] == available_classes.index(object_class))]

    # Draw rectangles and annotate the image
    if len(pred) > 0:
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for det in pred:
            box = det[:4].cpu().numpy()
            confidence = det[4].cpu().numpy()

            # Use the selected object class index
            class_index = available_classes.index(object_class)

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

        # Convert the Matplotlib figure to a base64-encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return image_data
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        object_class = request.form.get('object_class', '')  # Get the selected object class

        if file.filename == '' or object_class == '' or object_class not in available_classes:
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform object detection on the uploaded image with the selected object class
            image_data = detect_objects(file_path, object_class, confidence_threshold=0.5)

    return render_template('index.html', image_data=image_data, available_classes=available_classes)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
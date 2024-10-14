**YOLOv5 Object Detection**

This project implements object detection using the YOLOv5 model for both images and videos. The application allows users to upload media files (images or videos), and it performs real-time object detection, identifying multiple object classes and drawing bounding boxes around detected objects.

**Features:**

Image Detection: Upload an image and detect objects from a list of predefined classes (e.g., person, car, dog, etc.).

Video Detection: Upload a video, and the app will process each frame, detecting objects in real-time.

Confidence Threshold: Filter detected objects based on a user-defined confidence threshold.

Dynamic Object Classes: Users can choose specific object classes (from the list of 80 available classes) to focus on during detection.

Visualization: The application visualizes the detections by drawing bounding boxes around objects and labeling them with the detected class and confidence score.

Save and Download: Save and download the results as processed images or videos.

**Technologies Used:**

Flask: Web framework to handle image/video upload and interact with the object detection model.

YOLOv5: Pretrained deep learning model for object detection.

OpenCV: Library for image and video processing.

Matplotlib: Used for visualizing image results in the web app.

Streamlit: Used for building the interactive user interface for object detection in real-time.

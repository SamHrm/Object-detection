import streamlit as st
from PIL import Image
from object_detection import detect_objects_image, detect_objects_video
import os

st.title("Object Detection")

uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "mp4"])

confidence_threshold = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.5)

temp_file_path = None  # Initialize the variable

show_result_video = False  # Flag to control the visibility of the result video

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_file"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Run object detection
    if uploaded_file.type == "image/jpeg":
        st.image(Image.open(temp_file_path), caption="Original Image", use_column_width=True)
        if st.button("Run Object Detection"):
            with st.spinner("Running Object Detection..."):
                detect_objects_image(temp_file_path, confidence_threshold)
                st.image("output_image.png", caption="Processed Image", use_column_width=True)
    elif uploaded_file.type == "video/mp4":
        st.video(temp_file_path)
        if st.button("Run Object Detection"):
            with st.spinner("Running Object Detection..."):
                detect_objects_video(temp_file_path, confidence_threshold)
            show_result_video = True  # Set the flag to True after object detection

if temp_file_path and show_result_video:
    # Display the result video only when the flag is True and temp_file_path is defined
    st.video("output_video.mp4", format="video/mp4", start_time=0)

# Clean up temporary files
if temp_file_path and os.path.exists(temp_file_path):
    os.remove(temp_file_path)
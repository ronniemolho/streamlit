
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = '/content/drive/MyDrive/Cars Sheba/model_- 5 march 2024 11_47.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Object Detection using YOLOv8")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    uploaded_image = None  # Initialize uploaded_image variable
    if source_img:
        # Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, 1)
        # Resize the image to 640x640
        uploaded_image_resized = cv2.resize(uploaded_image, (640, 640))
        # Convert the image to RGB (OpenCV reads images in BGR format)
        uploaded_image_rgb = cv2.cvtColor(uploaded_image_resized, cv2.COLOR_BGR2RGB)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image_rgb,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        if st.sidebar.button('Detect Objects'):
            try:
                model = YOLO(model_path)
                res = model.predict(uploaded_image_resized, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                with col2:
                    st.image(res_plotted,
                             caption='Detected Image',
                             use_column_width=True
                             )
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
            except Exception as ex:
                st.error(
                    f"Unable to load model. Check the specified path: {model_path}")
                st.error(ex)

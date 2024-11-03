# import streamlit as st
# import requests
# from PIL import Image
# import io
# import base64

# # Set the backend URL
# BACKEND_URL = "http://localhost:8080"

# # Title of the app
# st.title("Image Processing App")

# # Initialize session state for storing the image and outputs
# if 'uploaded_image' not in st.session_state:
#     st.session_state['uploaded_image'] = None
# if 'processed_image' not in st.session_state:
#     st.session_state['processed_image'] = None
# if 'outputs' not in st.session_state:
#     st.session_state['outputs'] = None

# # 1. Upload Image Button
# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# # Store the uploaded image in session state
# if uploaded_file is not None:
#     st.session_state['uploaded_image'] = uploaded_file

# # 2. Process Button
# if st.button("Process"):
#     if st.session_state['uploaded_image'] is not None:
#         # Read the image file
#         image_bytes = st.session_state['uploaded_image'].read()
#         files = {'image': (st.session_state['uploaded_image'].name, image_bytes, st.session_state['uploaded_image'].type)}
#         data = {
#             'threshold_confidence': 0.5,
#             'threshold_iou': 0.7
#         }
#         # Send request to backend /inference endpoint
#         response = requests.post(f"{BACKEND_URL}/inference", data=data, files=files)
#         if response.status_code == 200:
#             result = response.json()
#             # Decode the base64 image
#             image_base64 = result.get('image_base64')
#             if image_base64:
#                 image_data = base64.b64decode(image_base64)
#                 image = Image.open(io.BytesIO(image_data))
#                 st.session_state['processed_image'] = image
#             # Store outputs
#             st.session_state['outputs'] = result.get('outputs')
#         else:
#             st.error(f"Error: {response.text}")
#     else:
#         st.warning("Please upload an image first.")

# # Display the processed image
# if st.session_state['processed_image'] is not None:
#     st.image(st.session_state['processed_image'], caption='Processed Image with Bounding Boxes')

# # Display the outputs
# if st.session_state['outputs'] is not None:
#     st.subheader("Outputs")
#     st.json(st.session_state['outputs'])

# # 3. Reformat with LLM Button
# if st.button("Reformat with LLM"):
#     if st.session_state['outputs'] is not None:
#         # Convert outputs to text or the required format
#         text_to_reformat = str(st.session_state['outputs'])
#         data = {'text': text_to_reformat}
#         # Send request to backend /llm endpoint
#         response = requests.post(f"{BACKEND_URL}/llm", json=data)
#         if response.status_code == 200:
#             result = response.json()
#             print(result)
#             st.subheader("Reformatted Output")  
#             st.json(result)
#         else:
#             st.error(f"Error: {response.text}")
#     else:
#         st.warning("Please process an image first.")



import streamlit as st
import requests
from PIL import Image
import io
import base64

# Set the backend URL
BACKEND_URL = "http://localhost:8080"

# Title of the app
st.title("Image Processing App")

# Initialize session state for storing the image and outputs
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'processed_image' not in st.session_state:
    st.session_state['processed_image'] = None
if 'outputs' not in st.session_state:
    st.session_state['outputs'] = None
if 'bounding_boxes' not in st.session_state:
    st.session_state['bounding_boxes'] = None

# 1. Upload Image Button
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Store the uploaded image in session state
if uploaded_file is not None:
    st.session_state['uploaded_image'] = uploaded_file

# 2. Process Button
if st.button("Process"):
    if st.session_state['uploaded_image'] is not None:
        # Read the image file
        image_bytes = st.session_state['uploaded_image'].read()
        files = {'image': (st.session_state['uploaded_image'].name, image_bytes, st.session_state['uploaded_image'].type)}
        data = {
            'threshold_confidence': 0.7,
            'threshold_iou': 0.7
        }
        # Send request to backend /inference endpoint
        response = requests.post(f"{BACKEND_URL}/inference", data=data, files=files)
        if response.status_code == 200:
            result = response.json()
            # Decode the base64 image
            image_base64 = result.get('image_base64')
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                st.session_state['processed_image'] = image
            # Store outputs and bounding boxes
            st.session_state['outputs'] = result.get('outputs')
            st.session_state['bounding_boxes'] = result['outputs'].get('bounding_boxes', [])
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please upload an image first.")

# Display the processed image
if st.session_state['processed_image'] is not None:
    st.image(st.session_state['processed_image'], caption='Processed Image with Bounding Boxes')

# Display the outputs
if st.session_state['outputs'] is not None:
    st.subheader("Outputs")
    # Exclude bounding_boxes from outputs when displaying
    outputs_without_boxes = st.session_state['outputs'].copy()
    outputs_without_boxes.pop('bounding_boxes', None)
    st.json(outputs_without_boxes)

# Display bounding box data
if st.session_state['bounding_boxes']:
    st.subheader("Bounding Boxes")
    for bbox in st.session_state['bounding_boxes']:
        st.write(f"Text: {bbox['text']}")
        st.write(f"Coordinates: {bbox['box']}")

# 3. Reformat with LLM Button
if st.button("Reformat with LLM"):
    if st.session_state['outputs'] is not None:
        # Convert outputs to text or the required format
        text_to_reformat = str(st.session_state['outputs'])
        data = {'text': text_to_reformat}
        # Send request to backend /llm endpoint
        response = requests.post(f"{BACKEND_URL}/llm", json=data)
        if response.status_code == 200:
            result = response.json()
            st.subheader("Reformatted Output")
            st.json(result)
        else:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please process an image first.")

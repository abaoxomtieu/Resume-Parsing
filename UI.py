import streamlit as st
import requests
from PIL import Image
import io
import base64
import json

# Set the backend URL
# BACKEND_URL = "https://abao77-cv-parsing.hf.space"
BACKEND_URL = "http://localhost:8080"

# Initialize session state
def init_session_state():
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    if "processed_image" not in st.session_state:
        st.session_state["processed_image"] = None
    if "extracted_text" not in st.session_state:
        st.session_state["extracted_text"] = None
    if "reformatted_cv" not in st.session_state:
        st.session_state["reformatted_cv"] = None
    if "bounding_boxes" not in st.session_state:
        st.session_state["bounding_boxes"] = None

def main():
    st.set_page_config(page_title="Resume Analyzer", layout="wide")
    init_session_state()
    
    # Title and description
    st.title("📄 Resume Analyzer & Job Matcher")
    st.markdown("""
    Upload your resume image, extract text, and match it with job descriptions.
    """)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Extract Text", "📝 Format Resume", "🎯 Job Matching"])

    # Tab 1: Extract Text
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Resume")
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
            
            if uploaded_file:
                st.session_state["uploaded_image"] = uploaded_file
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if st.button("📷 Extract Text", type="primary"):
                if st.session_state["uploaded_image"]:
                    with st.spinner("Processing image..."):
                        image_bytes = st.session_state["uploaded_image"].read()
                        files = {
                            "image": (
                                st.session_state["uploaded_image"].name,
                                image_bytes,
                                st.session_state["uploaded_image"].type,
                            )
                        }
                        data = {"threshold_confidence": 0.7, "threshold_iou": 0.7}
                        
                        try:
                            response = requests.post(f"{BACKEND_URL}/inference", data=data, files=files)
                            if response.status_code == 200:
                                result = response.json()
                                # Store processed image
                                image_base64 = result.get("image_base64")
                                if image_base64:
                                    image_data = base64.b64decode(image_base64)
                                    image = Image.open(io.BytesIO(image_data))
                                    st.session_state["processed_image"] = image
                                st.session_state["extracted_text"] = result.get("outputs")
                                st.success("Text extracted successfully!")
                            else:
                                st.error(f"Error: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please upload an image first.")

        with col2:
            if st.session_state["processed_image"]:
                st.subheader("Processed Image")
                st.image(st.session_state["processed_image"], caption="With Detected Text")

            if st.session_state["extracted_text"]:
                with st.expander("View Extracted Text", expanded=True):
                    st.json(st.session_state["extracted_text"])

    # Tab 2: Format Resume
    with tab2:
        st.subheader("Format Resume")
        if st.session_state["extracted_text"] is None:
            st.warning("Please extract text from an image first.")
        else:
            if st.button("🔄 Format Resume", type="primary"):
                with st.spinner("Formatting resume..."):
                    try:
                        data = {"text": str(st.session_state["extracted_text"])}
                        response = requests.post(f"{BACKEND_URL}/reformat_output", json=data)
                        if response.status_code == 200:
                            st.session_state["reformatted_cv"] = response.json()
                            st.success("Resume formatted successfully!")
                            
                            # Display formatted sections
                            col1, col2 = st.columns(2)
                            with col1:
                                with st.expander("📋 Personal Information", expanded=True):
                                    if "contact" in st.session_state["reformatted_cv"]:
                                        st.write(st.session_state["reformatted_cv"]["contact"])
                                
                                with st.expander("💼 Experience", expanded=True):
                                    if "experience" in st.session_state["reformatted_cv"]:
                                        st.write(st.session_state["reformatted_cv"]["experience"])
                            
                            with col2:
                                with st.expander("🎯 Skills", expanded=True):
                                    if "skills" in st.session_state["reformatted_cv"]:
                                        st.write(st.session_state["reformatted_cv"]["skills"])
                                
                                with st.expander("🌟 Other Information", expanded=True):
                                    if "interests" in st.session_state["reformatted_cv"]:
                                        st.write("Interests:", st.session_state["reformatted_cv"]["interests"])
                                    if "community" in st.session_state["reformatted_cv"]:
                                        st.write("Community:", st.session_state["reformatted_cv"]["community"])
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Tab 3: Job Matching
    with tab3:
        st.subheader("Match with Job Description")
        
        job_description = st.text_area("📝 Paste Job Description", height=200)
        
        if st.button("🎯 Match Resume", type="primary"):
            if job_description:
                if st.session_state["reformatted_cv"] is None:
                    st.warning("Formatting resume first...")
                    # Auto-format if not already formatted
                    data = {"text": str(st.session_state["extracted_text"])}
                    response = requests.post(f"{BACKEND_URL}/reformat_output", json=data)
                    if response.status_code == 200:
                        st.session_state["reformatted_cv"] = response.json()
                
                # Proceed with job matching
                with st.spinner("Analyzing match..."):
                    try:
                        data = {
                            "text": json.dumps(st.session_state["reformatted_cv"]),
                            "job_desciption": job_description
                        }
                        response = requests.post(f"{BACKEND_URL}/matching_job_desciption", json=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display match results
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.metric("Match Score", f"{result['score']}%")
                            
                            with col2:
                                st.markdown("### Match Analysis")
                                st.write(result["reasoning"])
                            
                            with st.expander("View Detailed Analysis", expanded=False):
                                st.json(result)
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a job description.")

if __name__ == "__main__":
    main()
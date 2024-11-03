from dotenv import load_dotenv
from src.utils.utils_segment import extract_text, draw_bounding_boxes
from fastapi import FastAPI, UploadFile, status, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from src.config.llm import llm
from src.prompt.promt import prompt_experience, format_prompt
from langchain_core.output_parsers import JsonOutputParser
import uvicorn
from io import BytesIO
import base64
from pydantic import Field, BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import functools
import threading
from src.inference.segment_inference import inference
from PIL import Image
load_dotenv()
app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor(max_workers=int(os.cpu_count() + 4))
parser = JsonOutputParser()


def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        thread_id = threading.get_ident()
        print(f"[Running function '{func_name}' in thread ID: {thread_id}]")
        return func(*args, **kwargs)

    return loop.run_in_executor(executor, functools.partial(wrapper, *args, **kwargs))


# def predict_func(threshold_confidence, threshold_iou, image):

#     image = np.frombuffer(image, np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     outputs = inference(
#         image,
#         threshold_confidence=threshold_confidence,
#         threshold_iou=threshold_iou,
#     )
#     text = extract_text(outputs=outputs, image_origin=image)
#     image = draw_bounding_boxes(image, outputs)
#     buffer = BytesIO()
#     image.save(buffer, format="JPEG")
#     buffer.seek(0)

#     image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#     response = {"outputs": text, "image_base64": image_base64}
#     return response

# def predict_func(threshold_confidence, threshold_iou, image):

#     image = np.frombuffer(image, np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     outputs = inference(
#         image,
#         threshold_confidence=threshold_confidence,
#         threshold_iou=threshold_iou,
#     )
#     text = extract_text(outputs=outputs, image_origin=image)
    
#     # Extract bounding boxes and associate them with the extracted text
#     # Assuming `outputs` contains bounding box data
#     bounding_boxes = []
#     for output in outputs:
#         box = output['box']  # Adjust according to your data structure
#         text_content = output['text']  # Adjust according to your data structure
#         bounding_boxes.append({
#             'box': box,  # e.g., [x_min, y_min, x_max, y_max]
#             'text': text_content
#         })

#     experience = text.get("experience", None)
#     chain = prompt_experience | llm | parser
#     experience_extracted = chain.invoke({"user_input": experience})
#     text.pop("experience", None)
#     promt_format_cv = format_prompt.format(input=text)
#     cv_output = llm.invoke(promt_format_cv)
#     cv_output_response = parser.parse(cv_output)
#     cv_output_response["experience"] = experience_extracted
    
#     # Add bounding boxes to the output
#     cv_output_response["bounding_boxes"] = bounding_boxes

#     # Convert image to PIL Image for saving
#     image_with_boxes = draw_bounding_boxes(image, outputs)
#     print(f"Type of image_with_boxes: {type(image_with_boxes)}")  # Debug statement

#     # Check the type of image_with_boxes and handle accordingly
#     if isinstance(image_with_boxes, np.ndarray):
#         # OpenCV image (NumPy array), convert color space
#         image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(image_rgb)
#     elif isinstance(image_with_boxes, Image.Image):
#         # PIL Image, no need to convert
#         image_pil = image_with_boxes
#     else:
#         # Unsupported type
#         raise TypeError(f"Unsupported type for image_with_boxes: {type(image_with_boxes)}")

#     buffer = BytesIO()
#     image_pil.save(buffer, format="JPEG")
#     buffer.seek(0)

#     image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
#     response = {"outputs": cv_output_response, "image_base64": image_base64}
#     return response


def arrange_bounding_boxes(outputs, order):
    """
    Arrange bounding boxes according to the specified order.

    Args:
        outputs: List of detection outputs containing 'box', 'text', and 'label' for each detection.
        order: List of section names in the desired order.

    Returns:
        List of bounding boxes ordered according to the specified sections.
    """
    # Create a dictionary to hold bounding boxes for each section
    section_bboxes = {section: [] for section in order}

    for output in outputs:
        box = output['box']  # [x_min, y_min, x_max, y_max]
        text_content = output['text']
        label = output.get('label', '').lower()

        # Assign the bounding box to the appropriate section
        for section in order:
            if section.lower() in label:
                section_bboxes[section].append({
                    'box': box,
                    'text': text_content
                })
                break
        else:
            # If no matching section, you can choose to handle it or skip
            pass

    # Flatten the bounding boxes into a list in the desired order
    ordered_bboxes = []
    for section in order:
        ordered_bboxes.extend(section_bboxes[section])

    return ordered_bboxes


def predict_func(threshold_confidence, threshold_iou, image):

    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    outputs = inference(
        image,
        threshold_confidence=threshold_confidence,
        threshold_iou=threshold_iou,
    )
    text = extract_text(outputs=outputs, image_origin=image)
    
    # Extract individual sections from the text
    contact = text.get("contact", "")
    profile = text.get("profile", "")
    skills = text.get("skills", "")
    community = text.get("community", "")
    education = text.get("education", "")
    experience = text.get("experience", "")
    interest = text.get("interest", "")

    # Process the 'experience' section with LLM
    chain = prompt_experience | llm | parser
    experience_extracted = chain.invoke({"user_input": experience})

    # Build the output dictionary in the specified order
    cv_output_response = {
        "contact": contact,
        "profile": profile,
        "skills": skills,
        "community": community,
        "education": education,
        "experience": experience_extracted,
        "interest": interest
    }
    
    # Add bounding boxes to the output
    cv_output_response["bounding_boxes"] = arrange_bounding_boxes(outputs, [
        "contact", "profile", "skills", "community", "education", "experience", "interest"
    ])

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image, outputs)
    # Handle image conversion
    if isinstance(image_with_boxes, np.ndarray):
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
    elif isinstance(image_with_boxes, Image.Image):
        image_pil = image_with_boxes
    else:
        raise TypeError(f"Unsupported image type: {type(image_with_boxes)}")

    # Encode image to base64
    buffer = BytesIO()
    image_pil.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = {"outputs": cv_output_response, "image_base64": image_base64}
    return response




@app.post("/inference", status_code=status.HTTP_200_OK)
async def predict(
    threshold_confidence: float = Form(default=0.7, ge=0, le=1),
    threshold_iou: float = Form(default=0.7, ge=0, le=1),
    image: UploadFile = File(...),
):
    try:
        image = await image.read()

        response = await run_in_thread(
            predict_func, threshold_confidence, threshold_iou, image
        )
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except Exception as e:
        response = {"error": str(e)}
        return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)


class LLMRequest(BaseModel):
    text: str = Field(..., title="Text to generate completion")


def call_llm(data):
    input = format_prompt.format(input=data)
    response = llm.invoke(input)
    response = JsonOutputParser().parse(response)
    return response


@app.post("/llm", status_code=status.HTTP_200_OK)
async def llm_predict(data: LLMRequest):
    try:
        response = await run_in_thread(call_llm, data.text)
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except Exception as e:
        response = {"error": str(e)}
        return JSONResponse(content=response, status_code=status.HTTP_400_BAD_REQUEST)


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8080, reload=True)

from dotenv import load_dotenv
from src.utils.utils_segment import extract_text, draw_bounding_boxes
from fastapi import FastAPI, UploadFile, status, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from src.config.llm import llm
from src.prompt.promt import format_prompt
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


def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        thread_id = threading.get_ident()
        print(f"[Running function '{func_name}' in thread ID: {thread_id}]")
        return func(*args, **kwargs)

    return loop.run_in_executor(executor, functools.partial(wrapper, *args, **kwargs))


def predict_func(threshold_confidence, threshold_iou, image):

    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    outputs = inference(
        image,
        threshold_confidence=threshold_confidence,
        threshold_iou=threshold_iou,
    )
    text = extract_text(outputs=outputs, image_origin=image)
    image = draw_bounding_boxes(image, outputs)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = {"outputs": text, "image_base64": image_base64}
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

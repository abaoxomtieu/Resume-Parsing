from langgraph.graph import StateGraph, END, START, add_messages
from typing import TypedDict, Any, Annotated
from PIL import Image
from src.utils.utils_segment import (
    preprocess,
    postprocess,
    extract_text,
    draw_bounding_boxes,
)
from src.inference.segment_inference import model
from src.config.llm import llm
from src.prompt.promt import format_prompt
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()


class State(TypedDict):
    image: Any
    image_origin: Any
    outputs_from_inference: Any
    text_extracted_from_ocr: Any
    threshold_confidence: float
    threshold_iou: float
    cropped_images: Any
    parser_output: bool
    image_with_bounding_boxes: Any
    _image: Annotated[Any, add_messages]
    crop_image: Any


class N:
    PRE_PROCESS = "PRE_PROCESS"
    POST_PROCESS = "POST_PROCESS"
    INFERENCE = "INFERENCE"
    EXTRACT_TEXT_FROM_OCR = "EXTRACT_TEXT_FROM_OCR"
    PARSER_WITH_LLM = "PARSER_WITH_LLM"
    IMAGE_WITH_BOUNDING_BOXES = "IMAGE_WITH_BOUNDING_BOXES"


workflow = StateGraph(State)


def pre_process_fn(state: State):
    preprocess_img = preprocess(state["image_origin"])
    print("preprocess_img", preprocess_img.shape)
    image_for_display = (preprocess_img[0] * 255).astype("uint8")
    image_for_display = image_for_display.transpose(1, 2, 0)
    image_show = Image.fromarray(image_for_display)
    return {"image": preprocess_img, "_image": image_show}


def inference_fn(state: State):
    image = state["image"]
    outputs = model.run(None, {"images": image})
    return {"outputs_from_inference": outputs}


def post_process_fn(state: State):
    outputs = state["outputs_from_inference"]
    threshold_confidence = state["threshold_confidence"]
    threshold_iou = state["threshold_iou"]
    post_process_output = postprocess(outputs, threshold_confidence, threshold_iou)

    return {
        "outputs_from_inference": post_process_output,
    }


def extract_text_from_ocr_fn(state: State):
    image_origin = state["image_origin"]
    output_from_inference = state["outputs_from_inference"]

    text = extract_text(output_from_inference, image_origin)
    return {"text_extracted_from_ocr": text}


def draw_bounding_boxes_fn(state: State):
    image = state["image_origin"]
    outputs = state["outputs_from_inference"]
    image_with_bounding_boxes = draw_bounding_boxes(image, outputs)
    return {"image_with_bounding_boxes": image_with_bounding_boxes}


def parser_output_fn(state: State):
    text_extracted_from_ocr = state["text_extracted_from_ocr"]
    chain = format_prompt | llm | parser
    response = chain.invoke({"user_input": text_extracted_from_ocr})
    print(response)
    return {"parser_output": response}

#NODE
workflow.add_node(N.PRE_PROCESS, pre_process_fn)
workflow.add_node(N.INFERENCE, inference_fn)
workflow.add_node(N.POST_PROCESS, post_process_fn)
workflow.add_node(N.EXTRACT_TEXT_FROM_OCR, extract_text_from_ocr_fn)
workflow.add_node(N.IMAGE_WITH_BOUNDING_BOXES, draw_bounding_boxes_fn)
workflow.add_node(N.PARSER_WITH_LLM, parser_output_fn)

#EDGE
workflow.add_edge(START, N.PRE_PROCESS)
workflow.add_edge(N.PRE_PROCESS, N.INFERENCE)
workflow.add_edge(N.INFERENCE, N.POST_PROCESS)
workflow.add_edge(N.POST_PROCESS, N.IMAGE_WITH_BOUNDING_BOXES)
workflow.add_edge(N.IMAGE_WITH_BOUNDING_BOXES, N.EXTRACT_TEXT_FROM_OCR)
workflow.add_conditional_edges(
    N.EXTRACT_TEXT_FROM_OCR,
    lambda state: N.PARSER_WITH_LLM if state["parser_output"] else END,
    {
        N.PARSER_WITH_LLM: N.PARSER_WITH_LLM,
        END: END,
    },
)
workflow.add_edge(N.PARSER_WITH_LLM, END)


app = workflow.compile()

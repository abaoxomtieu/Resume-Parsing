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
from src.prompt.promt import format_prompt, matching_jd_prompt
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

from typing_extensions import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages


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
    cv_matching_response: Any
    job_description: Any
    messages: Annotated[Sequence[Any], add_messages]


class N:
    PRE_PROCESS = "PRE_PROCESS"
    POST_PROCESS = "POST_PROCESS"
    INFERENCE = "INFERENCE"
    EXTRACT_TEXT_FROM_OCR = "EXTRACT_TEXT_FROM_OCR"
    PARSER_WITH_LLM = "FORMAT_OUTPUT"
    IMAGE_WITH_BOUNDING_BOXES = "IMAGE_WITH_BOUNDING_BOXES"
    CV_MATCHING = "CV_MATCHING"


workflow = StateGraph(State)


def pre_process_fn(state: State):
    preprocess_img = preprocess(state["image_origin"])
    print("preprocess_img", preprocess_img.shape)
    image_for_display = (preprocess_img[0] * 255).astype("uint8")
    image_for_display = image_for_display.transpose(1, 2, 0)
    image_show = Image.fromarray(image_for_display)
    image_show.show()
    message = HumanMessage("preprocess image")
    return {"image": preprocess_img, "_image": image_show, "messages": message}


def inference_fn(state: State):
    image = state["image"]
    outputs = model.run(None, {"images": image})
    message = HumanMessage("inference done")
    return {"outputs_from_inference": outputs, "messages": message}


def post_process_fn(state: State):
    outputs = state["outputs_from_inference"]
    threshold_confidence = state["threshold_confidence"]
    threshold_iou = state["threshold_iou"]
    post_process_output = postprocess(outputs, threshold_confidence, threshold_iou)
    message = HumanMessage("post process done")
    return {
        "outputs_from_inference": post_process_output, "messages": message
    }


from langchain_core.messages import HumanMessage, AIMessage


def extract_text_from_ocr_fn(state: State):
    image_origin = state["image_origin"]
    output_from_inference = state["outputs_from_inference"]

    text = extract_text(output_from_inference, image_origin)
    if not isinstance(text, str):
        text_ai = str(text)
    return {"text_extracted_from_ocr": text, "messages": AIMessage(text_ai)}


def draw_bounding_boxes_fn(state: State):
    image = state["image_origin"]
    outputs = state["outputs_from_inference"]
    image_with_bounding_boxes = draw_bounding_boxes(image, outputs)
    image_with_bounding_boxes.show()
    return {"image_with_bounding_boxes": image_with_bounding_boxes}


def parser_output_fn(state: State):
    text_extracted_from_ocr = state["text_extracted_from_ocr"]
    chain = format_prompt | llm | parser
    response = chain.invoke({"user_input": text_extracted_from_ocr})
    if not isinstance(response, str):
        text_ai = str(response)
    message = AIMessage(text_ai)
    return {"parser_output": response, "messages": message}


def cv_matching_fn(state: State):
    job_description = state["job_description"]
    parser_output = state["parser_output"]
    chain = matching_jd_prompt | llm | parser
    response = chain.invoke(
        {"job_description": job_description, "resume_input": parser_output}
    )
    if not isinstance(response, str):
        text_ai = str(response)
    message = AIMessage(text_ai)
    return {"cv_matching_response": response, "messages": message}


# NODE
workflow.add_node(N.PRE_PROCESS, pre_process_fn)
workflow.add_node(N.INFERENCE, inference_fn)
workflow.add_node(N.POST_PROCESS, post_process_fn)
workflow.add_node(N.EXTRACT_TEXT_FROM_OCR, extract_text_from_ocr_fn)
workflow.add_node(N.IMAGE_WITH_BOUNDING_BOXES, draw_bounding_boxes_fn)
workflow.add_node(N.PARSER_WITH_LLM, parser_output_fn)
workflow.add_node(N.CV_MATCHING, cv_matching_fn)

# EDGE
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
workflow.add_conditional_edges(
    N.PARSER_WITH_LLM,
    lambda state: END if state.get("job_description", None) is None else N.CV_MATCHING,
    {
        N.CV_MATCHING: N.CV_MATCHING,
        END: END,
    },
)
workflow.add_edge(N.CV_MATCHING, END)

app = workflow.compile()

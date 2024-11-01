import onnxruntime as ort
from src.utils.utils_segment import preprocess, postprocess
import numpy as np

model_path = "./src/model/best.onnx"
model = ort.InferenceSession(
    model_path,
)


def inference(image: np.array, threshold_confidence=0.5, threshold_iou=0.7):
    input = preprocess(image)
    outputs = postprocess(
        model.run(None, {"images": input}),
        threshold_confidence=threshold_confidence,
        threshold_iou=threshold_iou,
    )

    return outputs


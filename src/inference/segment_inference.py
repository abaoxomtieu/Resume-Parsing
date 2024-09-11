import onnxruntime as ort
from src.utils.utils_segment import preprocess, postprocess
from PIL import Image


def inference(image_path, model_path, threshold_confidence=0.5, threshold_iou=0.7):
    model = ort.InferenceSession(model_path)
    input = preprocess(image_path)
    outputs = postprocess(
        model.run(None, {"images": input}),
        threshold_confidence=threshold_confidence,
        threshold_iou=threshold_iou,
    )

    return outputs


if __name__ == "__main__":
    model_path = "../model/segment.onnx"
    image_path = "../../test.jpg"
    print(inference(image_path, model_path))

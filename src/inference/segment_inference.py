import onnxruntime as ort
from src.utils.utils_segment import preprocess, postprocess
from PIL import Image


def inference(image_path, model_path,threshold=0.5):
    model = ort.InferenceSession(model_path)
    input = preprocess(image_path)
    outputs = postprocess(model.run(None, {"images": input}),threshold=threshold)
    img = Image.open(image_path)
    cropped_images = []
    for obj in outputs:
        x1, y1, x2, y2, label, prob= obj
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img = {
            "image": cropped_img,
            "label": label,
            "prob": prob,
        }
        cropped_images.append(cropped_img)
    return cropped_images



if __name__ == "__main__":
    model_path = "../model/segment.onnx"
    image_path = "./test.jpg"
    print(inference(image_path, model_path))

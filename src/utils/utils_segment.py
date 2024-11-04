from PIL import Image
import numpy as np
import cv2
from typing import Tuple

from pytesseract import pytesseract

# path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
path_to_tesseract = r"./src/Tesseract-OCR/tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract
class_names = [
    "Community",
    "Contact",
    "Education",
    "Experience",
    "Interests",
    "Profile",
    "Skills",
]
number_class_custom = int(len(class_names) + 4)
img_width, img_height = None, None
left = None
top = None
ratio = None


def preprocess(img: np.array, shape=(640, 640)) -> np.array:
    global img_width, img_height, left, top, ratio
    img, ratio, (left, top) = resize_and_pad(img, new_shape=shape)
    img_height, img_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640).astype("float32")
    img = img / 255.0
    return img


def extract_box(outputs):
    output0 = outputs[0]
    output1 = outputs[1]
    output0 = output0[0].transpose()
    output1 = output1[0]
    boxes = output0[:, 0:number_class_custom]
    masks = output0[:, number_class_custom:]
    output1 = output1.reshape(32, 160 * 160)
    output1 = output1.reshape(32, 160 * 160)
    masks = masks @ output1
    boxes = np.hstack([boxes, masks])
    return boxes


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_mask(row, box, img_width, img_height, threshold):
    mask = row.reshape(160, 160)
    mask = sigmoid(mask)
    mask = (mask > threshold).astype("uint8") * 255
    x1, y1, x2, y2 = box
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    img_mask = Image.fromarray(mask, "L")
    img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    mask = np.array(img_mask)
    return mask


def get_polygon(mask):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygon = [[contour[0][0], contour[0][1]] for contour in contours[0][0]]
    return polygon


def postprocess(outputs, threshold_confidence, threshold_iou):
    objects = []
    for row in extract_box(outputs):
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        prob = row[4:number_class_custom].max()
        if prob < threshold_confidence:
            continue
        class_id = row[4:number_class_custom].argmax()
        label = class_names[class_id]
        objects.append([x1, y1, x2, y2, label, prob])

    # apply non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while objects:
        obj = objects.pop(0)
        result.append(obj)
        objects = [
            other_obj for other_obj in objects if iou(other_obj, obj) < threshold_iou
        ]
    del objects

    cropped_images = [
        {
            "box": list(map(int, unpad_and_resize_boxes(obj[:4], ratio, left, top))),
            "label": obj[4],
            "prob": int(obj[5]),
        }
        for obj in result
    ]
    return cropped_images


def extract_text_dict(outputs):
    result_dict = {}
    for output in outputs:
        label = output.get("label").lower()
        text = output.get("text")
        if label in result_dict:
            result_dict[label] += " " + text
        else:
            result_dict[label] = text

    return result_dict


def extract_text(outputs, image_origin):
    for i in range(len(outputs)):
        image = crop_image(image_origin, outputs[i].get("box"))
        text = pytesseract.image_to_string(image)
        outputs[i].update({"text": text})
        if "text" in outputs[i]:
            outputs[i]["text"] += text
        else:
            outputs[i].update({"text": text})
    return extract_text_dict(outputs)


# from PIL import Image
# from io import BytesIO
# import requests
# import base64

# def convert_image_to_base64(image_np):
#     """Convert a NumPy array (image) to a base64-encoded string."""
#     # Convert NumPy array to PIL Image
#     image_pil = Image.fromarray(image_np)

#     # Save the PIL image to a buffer in PNG format
#     buffered = BytesIO()
#     image_pil.save(buffered, format="PNG")

#     # Encode the buffer content as base64
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def send_image_to_api(image_base64):
#     """Send the base64-encoded image to the FastAPI API and return the extracted text."""
#     url = "https://abao77-pytesseract.hf.space/extract_text/"  # Replace with your FastAPI server URL
#     payload = {"base64_image": image_base64}
#     response = requests.post(url, json=payload)

#     # Extract text from the response
#     if response.status_code == 200:
#         return response.json().get("extracted_text", "")
#     else:
#         return ""

# def extract_text(outputs, image_origin):
#     for i in range(len(outputs)):
#         # Crop the image using the bounding box
#         image = crop_image(image_origin, outputs[i].get("box"))

#         # Convert the cropped image to base64
#         image_base64 = convert_image_to_base64(image)

#         # Call the API to extract text from the image
#         text = send_image_to_api(image_base64)

#         # Update the "text" field in the outputs list
#         if "text" in outputs[i]:
#             outputs[i]["text"] += text
#         else:
#             outputs[i].update({"text": text})

#     return extract_text_dict(outputs)


def crop_image(image, box):

    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def resize_and_pad(
    image: np.array,
    new_shape: Tuple[int, int],
    padding_color: Tuple[int] = (144, 144, 144),
) -> np.array:
    h_org, w_org = image.shape[:2]
    w_new, h_new = new_shape
    padd_left, padd_right, padd_top, padd_bottom = 0, 0, 0, 0

    # Padding left to right
    if h_org >= w_org:
        img_resize = cv2.resize(image, (int(w_org * h_new / h_org), h_new))
        h, w = img_resize.shape[:2]
        padd_left = (w_new - w) // 2
        padd_right = w_new - w - padd_left
        ratio = h_new / h_org

    # Padding top to bottom
    if h_org < w_org:
        img_resize = cv2.resize(image, (w_new, int(h_org * w_new / w_org)))
        h, w = img_resize.shape[:2]
        padd_top = (h_new - h) // 2
        padd_bottom = h_new - h - padd_top
        ratio = w_new / w_org

    image = cv2.copyMakeBorder(
        img_resize,
        padd_top,
        padd_bottom,
        padd_left,
        padd_right,
        cv2.BORDER_CONSTANT,
        None,
        value=padding_color,
    )

    return image, ratio, (padd_left, padd_top)


def unpad_and_resize_boxes(boxes, ratio, left, top):

    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 4)
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, :4] /= ratio
    if len(boxes) == 1:
        return boxes.flatten().tolist()
    else:
        return boxes.tolist()


def draw_bounding_boxes(image, outputs, save_path="output_image.jpg"):
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()

    # Define a list of colors for the bounding boxes
    label_colors = {
        "Community": (0, 255, 0),
        "Contact": (0, 0, 255),
        "Education": (255, 128, 0),
        "Experience": (255, 0, 255),
        "Interests": (128, 128, 128),
        "Profile": (0, 0, 128),
        "Skills": (128, 0, 128),
    }

    # Draw each bounding box and text
    for output in outputs:
        box = output["box"]
        label = output["label"]

        # Get the color for the label
        color = label_colors.get(
            label, (255, 255, 255)
        )  # Default to white if label not found

        # Draw the bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        # Draw the label and text
        cv2.putText(
            image_with_boxes,
            f"{label}",
            (x1, y1 - 10),
            cv2.FONT_ITALIC,
            2,
            color,
            2,
        )
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image (numpy array) to a PIL image
    image_pil = Image.fromarray(image_with_boxes_rgb)
    image_pil.save(save_path, format="JPEG")
    return image_pil

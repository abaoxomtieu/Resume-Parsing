from PIL import Image
import numpy as np
import cv2

class_names = [
    "Certifications",
    "Community",
    "Contact",
    "Education",
    "Experience",
    "Interests",
    "Languages",
    "Name",
    "Profil",
    "Projects",
    "skills",
]
number_class_custom = int(len(class_names) + 4)
img_width, img_height = None, None


def preprocess(image_path):
    global img_width, img_height
    img = Image.open(image_path)
    img_width, img_height = img.size
    img = img.convert("RGB")
    img = img.resize((640, 640))
    input = np.array(img)
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640).astype("float32")
    input = input / 255.0
    return input


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


def postprocess(outputs, threshold):
    objects = []
    for row in extract_box(outputs):
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        prob = row[4:number_class_custom].max()
        if prob < 0.5:
            continue
        class_id = row[4:number_class_custom].argmax()
        label = class_names[class_id]
        # mask = get_mask(
        #     row[number_class_custom:25684],
        #     (x1, y1, x2, y2),
        #     img_width,
        #     img_height,
        #     threshold=threshold,
        # )
        # polygon = get_polygon(mask)
        # objects.append([x1, y1, x2, y2, label, prob, mask, polygon])
        objects.append([x1, y1, x2, y2, label, prob])

    # apply non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(objects) > 0:
        result.append(objects[0])
        objects = [object for object in objects if iou(object, objects[0]) < 0.7]

    return result

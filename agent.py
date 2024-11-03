from src.graph import app
from PIL import Image
import numpy as np

# Read the image using PIL
image_path = "./src/data/resume-for-fresher-template-281.jpg"
image_pil = Image.open(image_path)

# Convert the PIL image to a NumPy array
image_np = np.array(image_pil)
graph = app

input = {
    "image_origin": image_np,
    "threshold_confidence": 0.5,
    "threshold_iou": 0.5,
    "parser_output": True,
}
output = app.invoke(input)

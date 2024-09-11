from src.inference.segment_inference import inference
from PIL import Image 
from pytesseract import pytesseract 
output = inference("D:/FU/DAT/test.jpg", model_path="./src/model/segment.onnx")
print(output)
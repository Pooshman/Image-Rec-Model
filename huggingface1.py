from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'https://images.unsplash.com/photo-1561997896-49c20aba9404?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8ZGFpc3klMjBmbG93ZXJ8ZW58MHx8MHx8fDA%3D'

image = Image.open(requests.get(url, stream = True).raw)

image.show(image)

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted Flower Type: ", model.config.id2label[predicted_class_idx])
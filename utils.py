import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
import torch
import torchvision
from torchvision import transforms

loader = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])  # resize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).float()
    image = normalize(image).float()
    # this is for VGG, may not be needed for ResNet
    image = image.unsqueeze(0).to(device)
    return image
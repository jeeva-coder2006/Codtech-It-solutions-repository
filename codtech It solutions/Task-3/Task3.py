import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import vgg19
from torchvision.utils import save_image
import torchvision.models as models
import style_transfer_utils  # assume helper file for style/content loss

# Load content and style images
content_img = Image.open("content.jpg")
style_img = Image.open("style.jpg")

# Resize and transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

content_tensor = transform(content_img).unsqueeze(0)
style_tensor = transform(style_img).unsqueeze(0)

# Assume a style transfer function exists
from neural_style_transfer import run_style_transfer  # Replace with actual code

output = run_style_transfer(content_tensor, style_tensor)
save_image(output, "stylized_output.jpg")
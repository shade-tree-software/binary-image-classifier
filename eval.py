import torch
import torch.nn as nn
import sys
from PIL import Image
import torchvision.transforms as transforms

from basicCNN import model

if len(sys.argv) < 3:
    print(f'usage: {sys.argv[0]} <model params state dict file> <image file>')
    exit()

model.load_state_dict(torch.load(sys.argv[1]))

img = Image.open(sys.argv[2])
rgb_img = img.convert('RGB')

img_size = 128
eval_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)),
  transforms.ToTensor(),
])

transformed_image = eval_transform(rgb_img)

model.eval()
result = model(torch.unsqueeze(transformed_image, 0))

print(result)

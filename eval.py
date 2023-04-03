import torch
import torch.nn as nn
import sys
from PIL import Image
import torchvision.transforms as transforms

if len(sys.argv) < 3:
    print(f'usage: {sys.argv[0]} <model params state dict file> <image file>')
    exit()

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.5))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.5))
model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))
model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())
model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
model.add_module('flatten', nn.Flatten())
model.add_module('fc', nn.Linear(256,1))
model.add_module('sigmoid', nn.Sigmoid())

model.load_state_dict(torch.load(sys.argv[1]))

img = Image.open(sys.argv[2])
rgb_img = img.convert('RGB')

img_size = 64
eval_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)),
  transforms.ToTensor(),
])

transformed_image = eval_transform(rgb_img)

model.eval()
result = model(torch.unsqueeze(transformed_image, 0))

print(result)
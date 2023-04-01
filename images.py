import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

img_size = 64
train_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])

imgdir0 = pathlib.Path('/mnt/andrew/photo/other/misc')
imgdir1 = pathlib.Path('/mnt/andrew/photo/other/misc2')
files0 = [str(path) for path in imgdir0.glob('*.jpg')]
files1 = [str(path) for path in imgdir1.glob('*.jpg')]
print(f'{len(files0)} files0')
print(f'{len(files1)} files1')

labels0 = [0] * len(files0)
labels1 = [1] * len(files1)

class ImageDataset(Dataset):
  def __init__(self, files, labels, transform=None):
    self.files = files
    self.labels = labels
    self.transform = transform

  def __getitem__(self, index):
    img = Image.open(self.files[index]).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    label = self.labels[index]
    return img, label

  def __len__(self):
    return len(self.labels)

  def set_transform(self, transform):
    self.transform = transform

init_ds = ImageDataset(files0 + files1,  labels0 + labels1, train_transform)
test_size = int(0.1 * len(init_ds))
full_size = len(init_ds) - test_size
full_ds, test_ds = torch.utils.data.random_split(init_ds, [full_size, test_size])
batch_size = 8
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

fig = plt.figure(figsize=(10,6))
for i in range(6):
  ax = fig.add_subplot(2, 3, i+1)
  ax.set_xticks([])
  ax.set_yticks([])
  img = full_ds[i][0].numpy().transpose((1,2,0))
  label = full_ds[i][1]
  ax.imshow(img)
  ax.set_title(f'{label}', size=15)

plt.tight_layout()
plt.show()

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

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, batch_size, full_ds):
  loss_hist_train = [0] * num_epochs
  accuracy_hist_train = [0] * num_epochs
  loss_hist_valid = [0] * num_epochs
  accuracy_hist_valid = [0] * num_epochs
  for epoch in range(num_epochs):
    valid_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - valid_size
    train_ds, valid_ds = torch.utils.data.random_split(full_ds, [train_size, valid_size])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)
    model.train()
    for x_batch, y_batch in train_dl:
      pred = model(x_batch)[:, 0]
      loss = loss_fn(pred, y_batch.float())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_hist_train[epoch] += loss.item()*y_batch.size(0)
      is_correct = ((pred>=0.5).float() == y_batch).float()
      accuracy_hist_train[epoch] += is_correct.sum()
    loss_hist_train[epoch] /= len(train_dl.dataset)
    accuracy_hist_train[epoch] /= len(train_dl.dataset)

    model.eval()
    with torch.no_grad():
      for x_batch, y_batch in valid_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch.float())
        loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
        is_correct = ((pred>=0.5).float() == y_batch).float()
        accuracy_hist_valid[epoch] += is_correct.sum()
    loss_hist_valid[epoch] /= len(valid_dl.dataset)
    accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

    print(f'Epoch {epoch+1} accuracy: '
          f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
          f'{accuracy_hist_valid[epoch]:.4f}')
  return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

num_epochs = 30
hist = train(model, num_epochs, batch_size, full_ds)

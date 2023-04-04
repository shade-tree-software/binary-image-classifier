import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import sys

# set up transformations
img_size = 64
train_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])

class ImageDataset(Dataset):
  # Can take either a list of filenames and a list of labels or a dataframe.
  # Dataframe is expected to be based on the celebA list_attr_celeba.txt attribute
  # file but with only two columns.  The first column should be the filename and
  # the second column should be a single attribute.
  def __init__(self, files=None, labels=None, df=None, image_dir='', transform=None):
    self.files = files
    self.labels = labels
    self.df = df
    self.image_dir = image_dir
    self.transform = transform

  def __getitem__(self, index):
    if self.df is not None:
      file = self.image_dir + self.df.iloc[index, 0]
      label = 1 if self.df.iloc[index, 1] == 1 else 0
    else:
      file = self.image_dir + self.files[index]
      label = self.labels[index]
    img = Image.open(file)
    rgb_img = img.convert('RGB') # convert any b&w images to color
    if self.transform is not None:
      rgb_img = self.transform(rgb_img)
    return rgb_img, label

  def __len__(self):
    if self.df is not None:
      return len(df.index)
    else:
      return len(self.labels)

  def set_transform(self, transform):
    self.transform = transform

if len(sys.argv) > 1 and sys.argv[1] == '-f':
  # setup dataset based on two directories of image files, one directory for each class
  imgdir0 = pathlib.Path('/mnt/andrew/photo/other/misc')
  files0 = [str(path) for path in imgdir0.glob('*.jpg')]
  labels0 = [0] * len(files0)
  print(f'{len(files0)} files0: {files0[:3]}...')
  imgdir1 = pathlib.Path('/mnt/andrew/photo/other/misc2')
  files1 = [str(path) for path in imgdir1.glob('*.jpg')]
  labels1 = [1] * len(files1)
  print(f'{len(files1)} files1: {files1[:3]}...')
  full_ds = ImageDataset(files=files0 + files1,  labels=labels0 + labels1, transform=train_transform)
else:
  # setup dataset based on CelebA data parsed into a dataframe
  celeba_dir = '/home/awhamil/Dev/machine-learning-book/ch12/celeba/'
  attrs = celeba_dir + 'list_attr_celeba.txt'
  df = pd.read_csv(attrs, header=None, skiprows=[0,1], usecols=[0, 32], sep='\s+').iloc[:10000]
  full_ds = ImageDataset(df=df, image_dir=celeba_dir + 'img_align_celeba/', transform=train_transform)

# split into training set, validation set, and test set
valid_size = int(0.1 * len(full_ds))
test_size = int(0.1 * len(full_ds))
train_size = len(full_ds) - valid_size - test_size
train_ds, valid_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, valid_size, test_size])
batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

# display first 6 images in training set
fig = plt.figure(figsize=(10,6))
for i in range(6):
  ax = fig.add_subplot(2, 3, i+1)
  ax.set_xticks([])
  ax.set_yticks([])
  img = train_ds[i][0].numpy().transpose((1,2,0))
  label = train_ds[i][1]
  ax.imshow(img)
  ax.set_title(f'{label}', size=15)
plt.tight_layout()
plt.show()

# create the model
# note: pytorch automatically initializes the weights of built-in layers in the nn module
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

def train(model, num_epochs, train_dl, valid_dl):
  loss_fn = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_hist_train = [0] * num_epochs
  accuracy_hist_train = [0] * num_epochs
  loss_hist_valid = [0] * num_epochs
  accuracy_hist_valid = [0] * num_epochs
  for epoch in range(num_epochs):
    model.train() # setup model for training
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

    model.eval() # setup model for evaluation
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
          f'{accuracy_hist_train[epoch]:.4f}, val_accuracy: '
          f'{accuracy_hist_valid[epoch]:.4f}')
  return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

# train the model for specified number of epochs, then save the weights
num_epochs = 30
print(f'Training set size = {len(train_ds)}')
print(f'Validation set size = {len(valid_ds)}')
print(f'Training for {num_epochs} epochs with batch size = {batch_size}')
hist = train(model, num_epochs, train_dl, valid_dl)
torch.save(model.state_dict(), f'./model_state_dict_{int(time.time())}')

# display charts of loss and accuracy 
x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()

#evaluate the model on the test set
accuracy_test = 0
model.eval()
with torch.no_grad():
  for x_batch, y_batch in test_dl:
    pred = model(x_batch)[:,0]
    is_correct = ((pred>=0.5).float() == y_batch).float()
    accuracy_test += is_correct.sum()
accuracy_test /= len(test_dl.dataset)
print(f'Test set size = {len(test_ds)}')
print(f'Test accuracy: {accuracy_test:.4f}')


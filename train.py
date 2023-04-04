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

from basicCNN import model

IMG_SIZE = 128
CELEBA_MAX_IMAGES = 15000
CELEBA_ATTR_FILE = 'list_attr_celeba.txt'
CELEBA_ATTR_COL = 32
CELEBA_IMAGE_DIR = 'img_align_celeba/'
BATCH_SIZE = 32
TEST_RATIO = 0.1
NUM_EPOCHS = 30

# set up transformations
train_transform = transforms.Compose([
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

if len(sys.argv) == 4 and sys.argv[1] == '-f':
  # setup dataset based on two directories of image files, one directory for each class
  imgdir0 = pathlib.Path(sys.argv[2])
  files0 = [str(path) for path in imgdir0.glob('*.jpg')]
  labels0 = [0] * len(files0)
  print(f'Class 0, {len(files0)} files in {sys.argv[2]}')
  imgdir1 = pathlib.Path(sys.argv[3])
  files1 = [str(path) for path in imgdir1.glob('*.jpg')]
  labels1 = [1] * len(files1)
  print(f'Class 1, {len(files1)} files in {sys.argv[3]}')
  full_ds = ImageDataset(files=files0 + files1,  labels=labels0 + labels1, transform=train_transform)
elif len(sys.argv) == 2:
  # setup dataset based on CelebA data parsed into a dataframe
  celeba_dir = sys.argv[1]
  attrs = celeba_dir + CELEBA_ATTR_FILE
  full_df = pd.read_csv(attrs, header=None, skiprows=[0,1], usecols=[0, CELEBA_ATTR_COL], sep='\s+')
  df = full_df.iloc[:CELEBA_MAX_IMAGES]
  full_ds = ImageDataset(df=df, image_dir=celeba_dir + CELEBA_IMAGE_DIR, transform=train_transform)
else:
  print(f'usage: {sys.argv[0]} <celeba dir>\n'
        f'       {sys.argv[0]} -f <class 0 dir> <class 1 dir>')
  exit(0)

# split into training set, validation set, and test set
valid_size = int(TEST_RATIO * len(full_ds))
test_size = int(TEST_RATIO * len(full_ds))
train_size = len(full_ds) - valid_size - test_size
train_ds, valid_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, valid_size, test_size])
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False)
test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

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

def train(model, NUM_EPOCHS, train_dl, valid_dl):
  loss_fn = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_hist_train = [0] * NUM_EPOCHS
  accuracy_hist_train = [0] * NUM_EPOCHS
  loss_hist_valid = [0] * NUM_EPOCHS
  accuracy_hist_valid = [0] * NUM_EPOCHS
  for epoch in range(NUM_EPOCHS):
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
print(f'Training set size = {len(train_ds)}')
print(f'Validation set size = {len(valid_ds)}')
print(f'Training for {NUM_EPOCHS} epochs with batch size = {BATCH_SIZE}')
hist = train(model, NUM_EPOCHS, train_dl, valid_dl)
torch.save(model.state_dict(), f'./model_state_dict_{int(time.time())}')

# display charts of loss and accuracy 
x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 5))
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


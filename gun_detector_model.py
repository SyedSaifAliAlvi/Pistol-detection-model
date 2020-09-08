from zipfile import ZipFile
import glob
import pandas as pd
import numpy as np
import os
import gzip
import random
import torch
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear,ReLU,CrossEntropyLoss,Sequential,LeakyReLU,ELU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import urllib.request
from skimage import transform
from skimage.util import random_noise

urllib.request.urlretrieve('https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/BK4.zip','dataset.zip')

file = 'dataset.zip'
with ZipFile(file,'r') as zp:
  zp.printdir()
  print("Extracting all.....")
  zp.extractall()
  print("Done..")

name = []
numeric_counter=0
for path in glob.glob("BK4/*"):
  if path=='BK4/AAAPistol':
    name.append(path)
    numeric_counter+=1
    print(numeric_counter)
  else:
    name.append(path)
    numeric_counter+=1

data_file_path = "BK4/AAAPistol/*.jpg"
label_counter = []
for image in glob.glob(data_file_path):
  label_counter.append(1)

len(label_counter)

def resize(img):
  image = Image.open(img)
  image = image.resize((128,128),Image.ANTIALIAS)
  return image

def imageToNumpyArray(img):
  N_array = np.asarray(img)
  return N_array

import cv2
def toThreeChannel(image):
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img2 = np.zeros_like(img)
  img2[:,:,0] = gray
  img2[:,:,1] = gray
  img2[:,:,2] = gray
  cv2.imwrite(image, img2)

def convertImagesToArray(path):
  img_array = []
  for image in glob.glob(path):
    toThreeChannel(image)
    R_img = imageToNumpyArray(resize(image))
    img_array.append(R_img)
  return img_array

x_data=[]
x_data=(convertImagesToArray("BK4/AAAPistol/*.jpg")).copy()

print(len(x_data),len(label_counter))

def convertImagesToArray(Mpath,Im_array):
  img_array = Im_array.copy()
  for path in glob.glob(Mpath): 
    new_path =path+"/*.jpg"
    if new_path=="BK4/AAAPistol/*.jpg":
      continue
    else:
      for image in glob.glob(new_path):
        toThreeChannel(image)
        R_img = imageToNumpyArray(resize(image))
        img_array.append(R_img)
  return np.asarray(img_array)

X = convertImagesToArray("BK4/*",x_data)
len(X)

for i in range(784):
    label_counter.append(0)

len(label_counter)

X = X[:1568]
len(X)

y = np.asarray(label_counter)

print(len(X),len(y))

X=np.einsum('ijkl->iljk',X)
X.shape

y.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .171, random_state=37)

X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))

class DataPreparation(Dataset):

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def data_setter(batch,features,labels):
  ds = DataPreparation(X=features,y=labels)
  ds = DataLoader(ds, batch_size=batch, shuffle=True)
  steps=features.shape[0]/batch
  return ds

def for_back(x,y,model,loss_fn,optimizer,validation=False):

  x=Variable(x).float()
  y=Variable(y).float()
  
  if validation==True:
    pred=model(x)
    loss=loss_fn(pred,y.type(torch.LongTensor))

  
  else:
    pred=model(x)
    loss=loss_fn(pred,y.type(torch.LongTensor))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  
  return loss.item()

class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.MaxPool2d(3)
        )

        
        self.fc1 = nn.Linear(in_features=6400, out_features=4500)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=4500, out_features=4800)
        self.fc3 = nn.Linear(in_features=4800, out_features=2000)
        self.fc4 = nn.Linear(in_features=2000, out_features=1800)
        self.fc5 = nn.Linear(in_features=1800, out_features=1600)
        self.fc6 = nn.Linear(in_features=1600, out_features=1200)
        self.fc7 = nn.Linear(in_features=1200, out_features=800)
        self.fc8 = nn.Linear(in_features=800, out_features=2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.drop(out)
        out = F.relu(self.fc3(out))
        out = self.drop(out)
        out = F.relu(self.fc4(out))
        out = self.drop(out)
        out = F.relu(self.fc5(out))
        out = self.drop(out)
        out = F.relu(self.fc6(out))
        out = self.drop(out)
        out = F.relu(self.fc7(out))
        out = self.drop(out)
        out = self.fc8(out)

        return out

def model_creation(learning_rate):
  model = CNN()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
  return model,criterion,optimizer

def Model_run(num_epochs,X_train,y_train,X_test,y_test,learning_rate,batch,lossHistory=True):
  train_losses=[]
  val_losses=[]
  steps=X_train.shape[0]/batch
  val_batch=int(X_test.shape[0]/steps)
  model,loss_fun,optim=model_creation(learning_rate)
  ds1 = data_setter(batch,X_train,y_train)
  ds2 = data_setter(val_batch,X_test,y_test)

  for epoch in range(num_epochs):
    i=0
    for (x1,y1),(x2,y2) in zip(ds1,ds2):
      tr_los=for_back(x1,y1,model,loss_fun,optim,validation=False)
      train_losses.append(tr_los)
      test_los=for_back(x2,y2,model,loss_fun,optim,validation=True)
      val_losses.append(test_los)
      i+=1
      if((i+1)%5==0 and lossHistory==True):
        print(f'epoch {epoch+1}/{num_epochs},step {i+1}/{steps},Train loss= {tr_los}, Validation loss= {test_los}')

  return train_losses,val_losses,model

def plot_loss(train_los,val_los):
  plt.plot(train_los, label='Training loss')
  plt.plot(val_los,label='Validation loss')
  plt.legend()
  plt.show()

def AccuracyScore(x,y,model,validation=False):
  if(validation==True):
    x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y), requires_grad=False)
    pred = model((x.float()))
    final_pred = np.argmax(pred.data.numpy(), axis=1)
    print("Validation Accuracy =",accuracy_score(y, final_pred))
  else:
    x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y), requires_grad=False)
    pred = model((x.float()))
    final_pred = np.argmax(pred.data.numpy(), axis=1)
    print("Training Accuracy =",accuracy_score(y, final_pred)*100,"%")

def Final_result(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no):
  train_loss,validation_loss,Real_model=Model_run(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no,lossHistory=True)
  plot_loss(train_loss,validation_loss)
  torch.save(Real_model, 'CNNnet.pkl')
  AccuracyScore(X_train,y_train,Real_model)
  AccuracyScore(X_test,y_test,Real_model,validation=True)

batch_no=50
num_epochs=15
lr_rate=0.0001
Final_result(num_epochs,X_train,y_train,X_test,y_test,lr_rate,batch_no)
print('For Epochs=',num_epochs,'Learning rate=',lr_rate,'Batch size=',batch_no)
print('________________________________________________________________________________________________________________________')
print('________________________________________________________________________________________________________________________')

cnn1 = torch.load('CNNnet.pkl')

def convertImagesToArray(path):
  img_array = []
  for image in glob.glob(path):
    toThreeChannel(image)
    R_img = imageToNumpyArray(resize(image))
    img_array.append(R_img)
  return np.asarray(img_array)

thamer = "/content/Browning-Hi-Power-pistol.jpg"
mt1=convertImagesToArray(thamer)
mt1=np.einsum('ijkl->iljk',mt1)
mt1.shape
mt1=torch.from_numpy(mt1)
mt1 = Variable(mt1).float()
y_mt1 = cnn1(mt1)

output=np.argmax(y_mt1.data.numpy(), axis=1)

output[0]

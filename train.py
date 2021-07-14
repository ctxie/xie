from efficientnet_pytorch import EfficientNet
from torch import nn
model = EfficientNet.from_pretrained('efficientnet-b0')
feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=4,bias=True)

import torch
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import torchvision.models as models
model1 = models.densenet121(pretrained=True)
model1.classifier = nn.Sequential(nn.Linear(1024,4))
                                  # nn.ReLU(),
                                  # nn.Dropout(0.2),
                                  # nn.Linear(256,2),
                                  # nn.LogSoftmax(dim=1)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model1.to(device)

from torch import nn
from torchvision.datasets import ImageFolder
import torch.optim as optim
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
transform = transforms.Compose( # 只能对PIL图片进行裁剪
    [transforms.ToTensor(),]
)
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable

train_dir='/home/new/xct/train'
val_dir='/home/new/xct/val'
#unlabeled_train_dir='/home/new/xct/un'
train_folder_set = ImageFolder(train_dir, transform=transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.5),transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(1, 1))],p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
val_folder_set = ImageFolder(val_dir, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
#un_folder_set = ImageFolder(unlabeled_train_dir, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]))


toPIL = transforms.ToPILImage()
#拼接
train_loader = DataLoader(dataset=train_folder_set, batch_size=8, shuffle=True,drop_last=False)
test_loader= DataLoader(dataset=val_folder_set, batch_size=8, shuffle=False,drop_last=False)
#un_loader= DataLoader(dataset=un_folder_set, batch_size=8, shuffle=True,drop_last=False)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.001},
    {'params': model1.parameters(), 'lr': 0.001}
])


loss_sum=0
val_loss=0
val_acc=0
cnt=0
#
for epoch in range(300):
    model.train()
    model1.train()
    for batch_id,data in tqdm(enumerate(train_loader)):
        inputs,target=data

        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        outputs1=model1(inputs)
        loss0=criterion(outputs,target)
        loss1=criterion(outputs1,target)
        loss=0.5*loss0+0.5*loss1
        loss.backward()
        optimizer.step()


    model.eval()#close dropout and batchnorm
    model1.eval()
    loss_sum=0
    correct=0
    total=0
    with torch.no_grad():  #close grad
        for b,data in tqdm(enumerate(test_loader)):
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            outputs1=model1(images)
            f=torch.nn.Softmax(dim=1)
            outputs_f=f(outputs)+f(outputs1)
            loss1=criterion(outputs,labels)
            loss2=criterion(outputs1,labels)
            loss_sum=loss_sum+loss1.item()*0.5+loss2.item()*0.5
            _,predicted=torch.max(outputs_f.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    acc=100*correct/total
    print('epoch:%d    Accuracy in test set:%4.f %%  loss:%.4f'%(epoch,acc,loss_sum))

    if epoch==0:
        val_loss=loss_sum
        val_acc=acc


    else:
        if acc>val_acc:
            cnt=0
            val_acc=acc
            p='/home/new/xct/result/eff_dense_a/eff_dense'+str(epoch+1)+'_acc_'+str(val_acc)+'.pth'
            save_model = {'model': model,'model1': model1,}
            torch.save(save_model, p)
            
        else:
            cnt=cnt+1
    if cnt>30:
        break


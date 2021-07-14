from efficientnet_pytorch import EfficientNet
from torch import nn
# model = EfficientNet.from_pretrained('efficientnet-b0')
# feature = model._fc.in_features
# model._fc = nn.Linear(in_features=feature,out_features=4,bias=True)

import torch
import torchvision

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import torchvision.models as models
# model1 = models.densenet121(pretrained=True)
# model1.classifier = nn.Sequential(nn.Linear(1024,4))
                                  # nn.ReLU(),
                                  # nn.Dropout(0.2),
                                  # nn.Linear(256,2),
                                  # nn.LogSoftmax(dim=1)
save_name = '/home/new/xct/result/eff_dense_a/eff_dense100_acc_91.91919191919192.pth'
load_models = torch.load(save_name)
model = load_models['model']
model1= load_models['model1']
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model1.to(device)


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)








from torch import nn
from torchvision.datasets import ImageFolder
import torch
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




toPIL = transforms.ToPILImage()
#拼接

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)



model.eval()
model1.eval()
import csv
print('kaishi')
'''-------------------------------------------------------'''
csvFile = open("/home/new/xct/result/eff_dense_a/test_eff_dense__919_improve.csv", "w",encoding='utf-8',newline="")            #创建csv文件
writer = csv.writer(csvFile)                  #创建写的对象
#先写入columns_name
writer.writerow(["image","none","infection","ischaemia","both"])     #写入列的名称

#读入图片
test_root = '/home/new/xct/test'
img_test=os.listdir(test_root)
img_test.sort(key=lambda img_test:int(img_test.split('.')[0]))
trans=transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
for i in range(len(img_test)):
    #rd_img = cv2.imread(test_root+img_test[i])
    img = Image.open(os.path.join(test_root,img_test[i]))
    img = img.convert('RGB')

    input=trans(img)
    input=input.unsqueeze(0)#这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    #增加一维，输出的img格式为[1,C,H,W]
    input = input.to(device)

    score = model(input)#将图片输入网络得到输出
    probability = torch.nn.functional.softmax(score,dim=1)#计算softmax，即该图片属于各类的概率
    probability=probability.tolist()

    score1=model1(input)
    probability1 = torch.nn.functional.softmax(score1,dim=1)#计算softmax，即该图片属于各类的概率
    probability1=probability1.tolist()
    #max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别
    #class_index = result_(index)
    #probability=np.round(probability.cpu().detach().numpy(),3)
    a=(probability[0][3]+probability1[0][3])/2
    b=(probability[0][1]+probability1[0][1])/2
    c=(probability[0][2]+probability1[0][2])/2
    d=(probability[0][0]+probability1[0][0])/2
    writer.writerow([img_test[i],a,b,c,d])
csvFile.close()






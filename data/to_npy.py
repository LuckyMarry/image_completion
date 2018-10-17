#主要功能是读取数据-洗牌-分为train和test且储存为.npy结构

import glob
import os
import cv2
import numpy as np

ratio=0.95 #%95训练 %5测试
image_size=128 #输入大小

x=[]
x_1=[]
x_2=[]

paths=glob.glob('/Users/yunma/Downloads/data/img_align_celeba/*.jpg')
paths_1=glob.glob('/Users/yunma/Downloads/data/picture/*.jpg')
for path in paths[:500]:
    img=cv2.imread(path)
    img=cv2.resize(img,(image_size,image_size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x.append(img)

for i in paths_1:
    img=cv2.imread(i)
    img=cv2.resize(img,(image_size,image_size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x_1.append(img)

for i in paths[500:600]:
    img=cv2.imread(i)
    img=cv2.resize(img,(image_size,image_size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x_2.append(img)

x=np.array(x,dtype=np.uint8)#list 转换为np.array的结构
x_1=np.array(x_1,dtype=np.uint8)
x_2=np.array(x_2,dtype=np.uint8)

np.random.shuffle(x)
np.random.shuffle(x_1)
np.random.shuffle(x_2)


p=int(ratio*len(x))
x_train=x[:p]
x_test=x[p:]

if not os.path.exists('./npy'):
     os.mkdir('/npy')

np.save('npy/x_train.npy',x_train)
np.save('npy/x_test.npy',x_test)
np.save('npy/x_test_1.npy',x_1)
np.save('npy/x_test_2.npy',x_2)



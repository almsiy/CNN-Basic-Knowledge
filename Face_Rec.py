import os
import glob
import h5py
import keras
import numpy as np
from PIL import Image
from keras.models import load_model

from Name import *

def read_img(img_location):
    img = []
    x_s = 200
    y_s = 200
    #for pic in glob.glob(location+'\\'+'\*.png'): # 限制图像格式为*.png
    im = Image.open(img_location) # 读取图像数据
    im = im.resize((x_s, y_s), Image.ANTIALIAS) # 归一化
    im = np.array(im) # 将图像保存为numpy矩阵
    if(im.shape[0]==200 and im.shape[1]==200): 
        r = im[:,:,0] # 读取图像中各个像素点的 R 值
        g = im[:,:,1] # 读取图像中各个像素点的 G 值
        b = im[:,:,2] # 读取图像中各个像素点的 B 值
        img.append([r,g,b]) # 存储测试集
    return np.array(img)

if __name__ == '__main__':

    model = load_model('face.h5')

    img_rows = 200 # 高
    img_cols = 200 # 宽

    #for img_location in glob.glob('face\\Adam_Brody\\*.png'): # 限制图像格式为*.png
    for img_location in glob.glob('test\\*.png'): # 限制图像格式为*.png
        print('\nImage:',img_location)
        img = read_img(img_location)
        img = img.reshape(img.shape[0], img_rows, img_cols, 3) 
        img = img.astype('float32') 
        img /= 255
        pre = model.predict_classes(img) # 返回预测的标签值
        #pre = model.predict(img)
        for i in pre:
            print('Name:',Name.get(i))
        #print(pre)
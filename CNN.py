import os
import glob
import h5py
import keras
import numpy as np
from PIL import Image
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

def read_img(location):
    x_train = [] 
    y_train = [] 
    x_test = [] 
    y_test = [] 
    label_name = [] 
    dirs = os.listdir(location) 
    label = 0 
    count = 0

    for i in dirs: # 遍历文件夹
        print(i)
        n = 0 
        label_name.append(i) # 将文件夹的名称作为 保存 save folder name in var label_name
        x_s = 200
        y_s = 200
        for pic in glob.glob(location+'\\'+i+'\*.png'): # 限制图像格式为*.png
            im = Image.open(pic) # 读取图像数据
            im = im.resize((x_s, y_s), Image.ANTIALIAS) # 归一化
            im = np.array(im) # 将图像保存为numpy矩阵
            if(im.shape[0]==200 and im.shape[1]==200): 
                r = im[:,:,0] # 读取图像中各个像素点的 R 值
                g = im[:,:,1] # 读取图像中各个像素点的 G 值
                b = im[:,:,2] # 读取图像中各个像素点的 B 值
                if(n<30): # 每组取10张图片作为测试集
                    x_test.append([r,g,b]) # 存储测试集
                    y_test.append([label]) # 存储测试集标签
                    x_train.append([r,g,b]) # 存储训练集
                    y_train.append([label]) # 存储训练集标签
                else: # 其余图片作为训练集
                    x_train.append([r,g,b]) # 存储训练集
                    y_train.append([label]) # 存储训练集标签
                n = n + 1 
                count = count + 1 
        label = label + 1 # increment label
    print(label_name)
    #print(dirs)
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

#path='add_face'
path = 'face'
img_rows = 200 # 高
img_cols = 200 # 宽
num_class = 176 # 标签数量 num of classes/labels
x_train,y_train,x_test,y_test = read_img(path)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3) 
input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 
y_train = keras.utils.to_categorical(y_train, num_class) 
y_test = keras.utils.to_categorical(y_test, num_class)

print(x_train.shape) # 训练集数据维度
print(y_train.shape) # 训练集标签维度
print(x_test.shape) # 测试集数据维度
print(y_test.shape) # 测试集标签维度

model = Sequential()
#
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
#
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 全连接层
model.add(Dense(num_class, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

#model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data=(x_test, y_test))
#model.save('face.h5')

for i in range(50):
    print('The '+str(i)+' th Iteration')
    model=load_model('face.h5')
    model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data=(x_test, y_test))
    model.save('face.h5')
    json_string = model.to_json()
    open('face.json','w').write(json_string)
    model.save_weights('face_weights.h5')
    K.clear_session()
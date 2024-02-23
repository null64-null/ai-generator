import sys, os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from mnist.mnist import load_mnist

####### pkl -> data set #######
def unpickle(file_path):
    with open(file_path, 'rb') as fo: 
        picture_data_dict = pickle.load(fo, encoding='bytes')
    return picture_data_dict

def generate_image_matrix(images_data, image_shape):
    n, c, w, h = image_shape
    images = images_data.reshape(n, c, w, h)
    return images

def generate_data(picture_data_dict, image_shape):
    x = generate_image_matrix(picture_data_dict[b'data'], image_shape)
    t = np.array(picture_data_dict[b'labels'])
    filenames = picture_data_dict[b'filenames']
    
    return x, t, filenames

def generate_data_from_pkl(file_path, image_shape):
    picture_data_dict = unpickle(file_path)
    x, t, filenames = generate_data(picture_data_dict, image_shape)

    return x, t, filenames


####### data set -> pkl #######
def data_to_pkl(x, t, filenames, path):
    n, c, w, h = x.shape
    data = x.reshape(n, c*w*h)
    labels = t.tolist()
    
    dict = {
        b'data': data,
        b'labels': labels,
        b'filenames': filenames
    }

    with open(path, 'wb') as f:
        pickle.dump(dict, f)









'''
path = '/Users/yudaihamashima/Downloads/cifar-10-batches-py/data_batch_1'
save_path = '/Users/yudaihamashima/Downloads/cifar-10-batches-py/test.pkl'

x, t, filenames = generate_data_from_pkl(path, [10000, 3, 32, 32])

data_to_pkl(x, t, filenames, save_path)
x1, t1, filenames1 = generate_data_from_pkl(save_path, [10000, 3, 32, 32])

print(t1[3])
display_color_image(x1[3])
'''


'''
# fileはpickleかつバイナリ形式で保存されている、Pythonオブジェクトをflatteにするのによいらしい
# pickle形式はPythonオブジェクトを直列化（シリアライズ）して保存するためのフォーマット
# pickle形式では、キーがバイト文字列で表される
def unpickle(file):
    #'r'ead 'b'inaryということ、fileをバイナリ形式で開くということ
    with open(file, 'rb') as fo: 
        #dictはpickle形式の画像データ（バイト文字列になる）
        #複数のキーを持ち、画像に関する情報がいろいろ入っている。
        #encoding='bytes'はバイト文字列にするよということ
        dict = pickle.load(fo, encoding='bytes')
    return dict
'''

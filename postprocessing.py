import os
import os.path
from os import walk
from os.path import join
import numpy as np
from keras.models import load_model
from itertools import groupby
from PIL import Image
import time
import csv

base_path = '/home/workspace/kaggle/download_files/'


def get_picture(path):
    path0 = join(base_path + path)
    picture = []
    for _, _, filenames in walk(path0):
        for filename in filenames:
            file_prefix = os.path.splitext(filename)[0]
            if os.path.exists(join(path0, file_prefix + ".jpg")):
                picture.append(filename)
            elif os.path.exists(join(path0, file_prefix + ".gif")):
                picture.append(filename)
            else:
                print("路径不对或者没有这种格式的文件")
    return picture


def predict_from_model(patch, model):
    prediction = model.predict(patch.reshape(1, 128, 128, 3))
    prediction = prediction[:, :, :, 1].reshape(128, 128)
    return prediction


# 定义预测整张汽车图片的函数
def image_prediction(model, car_pic, widths=128, heights=128):
    pat_pred = np.zeros((128, 1920))
    for i in range(car_pic.size[1]//heights):
        pat_pre = np.zeros((128,128))
        for j in range(car_pic.size[0]//widths):
            pat = car_pic.crop((j*widths, i*widths, (j+1)*widths, (i+1)*heights))
            pat_array = np.array(pat)
            pre_pat = predict_from_model(pat_array, model)
            pat_pre = np.hstack((pat_pre, pre_pat))
        pat_pred = np.vstack((pat_pred, pat_pre))
#     print((np.array(pat_pred)).shape)
    predic01 = np.array(pat_pred)
    for k in range(widths):    # 去掉两行,1 pixel
        predic01 = np.delete(predic01, k, 1)
        predic01 = np.delete(predic01, k, 0)
    predic02 = np.column_stack((predic01, (np.zeros((1280, 126))).astype(int)))    # 增加一排,126 pixel
#     print(predic02.shape)
    return predic02


def pred_by_threshold(prediction, threshold):

    '''
    本函数通过设置阈值，归一化预测结果。

    输入：
        prediction:h5预测的结果，是一个概率矩阵
        threshold:阈值，取值范围是0-1之间
    输出：
        pred_bin:一维二值数组
        pred_one_index:一维角标数组，pred_bin非零元素的角标组成的数组
    '''

    pred_th = []
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j] > threshold:
                pred_th.append(1)
            else:
                pred_th.append(0)
    pred_bin = np.array(pred_th)
    pred_one_index = np.array(np.nonzero(pred_bin))
    return pred_bin, pred_one_index


def first_half(pred_one_index):
    '''
    操作角标的函数，将数组中比前一元素大1的元素挑出来，并且删除，eg：
        input: [1,2,3,4,9,11,15,17,18,19]
        output: [2,3,4,18,19]
        output2: [1,9,11,15,17]

    输入：
        预测矩阵的非零元素的角标数组
    输出：
        非连续的数字元素组成的新数组
    '''
    array_p = pred_one_index[0][:]
    re_index = []
    for index, item in enumerate(array_p):
        if index>0 and array_p[index]-array_p[index-1] == 1:
            re_index.append(index)
    fh = np.delete(array_p, re_index, 0)
    return fh


def RLE(data):
    x = []
    for name, group in groupby(data):
        if name == 1:
            x.append(len(list(group)))
    yield x


def second_half(pred_bin):
    second_half = next(RLE(pred_bin))
#     print(len(second_half))
    return second_half


def rle_mask(first_half, second_half):
    '''
    Input:
        first_half:[0,4,7,10,19]
        second_half:[3,1,2,3,7]
    Output:
        [0,3,4,1,7,2,10,3,19,7]
    '''
    rle_element = (np.ones((2*len(second_half)))).astype(int)    # 定义行程长度数组
#     print(rle_element,'\n')
    for i in range(first_half.shape[0]):
        rle_element[i*2] =  first_half[i]
        rle_element[i*2+1] = second_half[i]
#     print(rle_element,'\n')
    rle_mask = []
    for i in range(rle_element.shape[0]):
        rle_mask.append(rle_element[i])
    return rle_mask


model = load_model('./model_64_200.h5')
car_pictures = get_picture("train")
mask_pictures = get_picture("train_masks")

print(len(car_pictures), len(mask_pictures))

train_set = car_pictures[0:3801]
valid_set = car_pictures[3801:5068]
test_set = car_pictures[5068:5088]
print(len(train_set), '\n', len(valid_set), '\n', len(test_set))

# test = get_picture('test')
# print(test[0])

start_time = time.clock()

csvfile = open('submission.csv', 'w')
fieldnames = ['img', 'rle_mask']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
for i in range(len(test_set)):
    order_path = join(base_path, "train/" + test_set[i])
    car_pic = Image.open(order_path)
    image_pred = image_prediction(model, car_pic)
    pred_bin, pred_one_index = pred_by_threshold(image_pred, 0.5)
    first_h = first_half(pred_one_index)
#     print(len(first_h))
    second_h = second_half(pred_bin)
    rle_mask_value = rle_mask(first_h, second_h)
    writer.writerow({'img': test_set[i], 'rle_mask': (str(rle_mask_value)).replace(',', '').replace('[', '').replace(']', '')})
    if i-(i//2)*2 == 0:
        print('Picture %d is processing' % i)
csvfile.close()
end_time = time.clock()
print('Time consuming : %.2f s' % (end_time-start_time))

# importing the libraries
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import splitfolders
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pickle
import os
# Prediction as function

def predict(frame_path, data_write_name, model_path, output_save_path):

    images = None
    data_write_name = data_write_name
    path = frame_path
    images = glob.glob(path + "/*.jpg")
    print('No of images found in the folder: ', len(images))

    # load model
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # creating the dictionary of images to maintain the sequence and retrieve the data faster

    list_predictions = []
    list_frames = []

    dic = {}

    for image in images:
        file_name = image.split('\\')[-1]
        # frame = file_name.split('_')[0]
        frame = int(file_name.split('_')[0])
        # frame = int(frame.split('Cam2')[-1])
        dic[frame] = image

    prediction_raw = []

    for fr_no in tqdm(range(len(dic))):
        fr_no = fr_no + 1
        img = cv2.imread(dic[fr_no])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 500))
        img = img / 255
        img = tf.expand_dims(img, axis=0)
        result = model.predict(img)
        prediction_raw.append(result)
        list_predictions.append(result.argmax(axis=-1)[0])
        list_frames.append(fr_no)

    # Writing raw prediction
    file_path = output_save_path + "\\raw_predict_" + data_write_name + ".txt"
    if os.path.exists(file_path) == False:
        with open(file_path, 'w') as f:
            for item in prediction_raw:
                f.write("%s\n" % item)
    else:
        with open('temp_raw_predict.txt', 'w') as f:
            for item in prediction_raw:
                f.write("%s\n" % item)
        raise ValueError("The file already exist. Please provide another name writing raw prediction")

    # writing raw prediction for analysis
    file_path = output_save_path + "Data_raw_predict_" + data_write_name + ".txt"
    if os.path.exists(file_path) == False:
        with open(file_path, "wb") as fp:
            pickle.dump(prediction_raw, fp)
    else:
        with open('temp_Data_raw_predict.txt', "wb") as fp:
            pickle.dump(prediction_raw, fp)
        raise ValueError("The file already exist. Please provide another name writing Data raw prediction")


#
# # reading the images extracted from the video
# images = None
# data_write_name = "26_c_al"                                      ##### Need to changed
# path = 'D:\\16 Max Plank\\Dev\\Experiments\\raw_image_26_c_al\\' ##### Need to changed
#
#
# images = glob.glob(path+"/*.jpg")
# print('No of images found in the folder: ', len(images))
#
# # loading the model
# model = tf.keras.models.load_model('D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\model.hf5\\')
# model.summary()
#
# # creating the dictionary of images to maintain the sequence and retrieve the data faster.
# list_predictions = []
# list_frames = []
#
# dic = {}
#
# for image in images:
#     file_name = image.split('\\')[-1]
#     #frame = file_name.split('_')[0]
#     frame = int(file_name.split('_')[0])
#     # frame = int(frame.split('Cam2')[-1])
#     dic[frame] = image
#
# prediction_raw = []
#
# for fr_no in tqdm(range(len(dic))):
#     fr_no = fr_no + 1
#     img = cv2.imread(dic[fr_no])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (500, 500))
#     img = img/255
#     img = tf.expand_dims(img, axis=0)
#     result = model.predict(img)
#     prediction_raw.append(result)
#     list_predictions.append(result.argmax(axis=-1)[0])
#     list_frames.append(fr_no)
#
# # Writing raw prediction
# file_path = "D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\raw_predict_" + data_write_name +".txt"
# if os.path.exists(file_path) == False:
#     with open(file_path, 'w') as f:
#         for item in prediction_raw:
#             f.write("%s\n" % item)
# else:
#     with open('temp_raw_predict.txt', 'w') as f:
#         for item in prediction_raw:
#             f.write("%s\n" % item)
#     raise ValueError("The file already exist. Please provide another name writing raw prediction")
#
# # writing raw prediction for analysis
# file_path = "D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\Data_raw_predict_" + data_write_name +".txt"
# if os.path.exists(file_path) == False:
#     with open(file_path, "wb") as fp:
#         pickle.dump(prediction_raw, fp)
# else:
#     with open('temp_Data_raw_predict.txt', "wb") as fp:
#         pickle.dump(prediction_raw, fp)
#     raise ValueError("The file already exist. Please provide another name writing Data raw prediction")
#
# goosebump = []
# no_goosebump = []
# marker = []
#
# for i in tqdm(range(len(list_predictions))):
#     if list_predictions[i] == 0:  # goosebump
#         goosebump.append(1)
#         no_goosebump.append(0)
#         marker.append(0)
#
#     elif list_predictions[i] == 2:  # no_goosebump
#         goosebump.append(0)
#         no_goosebump.append(1)
#         marker.append(0)
#
#     elif list_predictions[i] == 1:  # marker
#         goosebump.append(0)
#         no_goosebump.append(0)
#         marker.append(1)

if __name__=='__main__':
    pass
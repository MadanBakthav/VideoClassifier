# Video to images

import cv2

def video_frames(video_path,frame_path):
    path = video_path #'D:\\16 Max Plank\\Dev\\Experiments\\26_c_Cam0_1912181631_arli.avi'
    vidcap = cv2.VideoCapture(path)
    video_name = path.split('\\')
    video_name = video_name[-1].split('_')
    video_name = str(video_name[0]) + str(video_name[1]) + str(video_name[2])
    success,image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(frame_path + str(count)+"_frame_"+str(round(count/15,2))+".jpg", image)     # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    vidcap.release()

# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# open_file = open("D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\Data_raw_predict_2_c_al.txt", "rb")
# loaded_list = pickle.load(open_file)
# open_file.close()
# #print(loaded_list)
#
# list_pred = np.asarray(loaded_list)
# goose = list_pred[...,0]
# plt.plot(goose)
# plt.show()
# print("end")#

if __name__=='__main__':
    pass
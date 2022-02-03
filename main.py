from data_prep import *
from postprocessing import *
from prediction_v3 import *
from tqdm import tqdm
# Data preparation
import os
video_input_folder = "D:\\16 Max Plank\\Dev\\Experiments\\"

# number = 60
# v1 = "13_c_Cam0_1908301222_arli"
v1 = "17_a_Cam0_1909031333_arli"
v2 = "17_c_Cam0_1909171208_arli"
#v4 = "29_c_Cam0_1911271549_arli"

list_videos = [v1, v2]
# list_videos = [v1]
list_file_names = [str(17)+"_a_al",str(17)+"_c_al"]
# list_file_names = [str(number)+"_a_al",str(number)+"_b_al",str(number)+"_c_al"]
output_save_path = "D:\\16 Max Plank\\Dev\\Experiments\\12_con_modularised\\"
model_output_save_path = "D:\\16 Max Plank\\Dev\\Experiments\\12_con_modularised\\final_analysis\\"
model_path = "D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\model.hf5\\"

for i in range(len(list_videos)):
    frame_path = output_save_path + list_file_names[i] + "\\"
    os.makedirs(frame_path)
    video_path = video_input_folder + list_videos[i] + ".avi"
    video_frames(video_path,frame_path)

    predict(frame_path, list_file_names[i], model_path,model_output_save_path)




print('end')


import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def write_to_file(file_path, data_to_file):
    print('Type of data coming in : ', type(data_to_file))
    if os.path.exists(file_path) == False:
        if type(data_to_file) != pd.DataFrame:
            with open(file_path, 'w') as f:
                for item in data_to_file:
                    f.write("%s\n" % item)

        elif type(data_to_file) == pd.DataFrame:
            numpy_array = data_to_file.to_numpy()
            np.savetxt(file_path, numpy_array, fmt="%s")


    else:
        raise ValueError("Fn write_list: The file already exist. Please provide another name")

def read_raw_prediction(file_path):

    open_file = open(file_path, "rb")
    raw_prediction_list = pickle.load(open_file)
    open_file.close()
    print('....Loaded raw prediction data successfully ....')
    return np.array(raw_prediction_list)


def get_graph(class_name, raw_prediction_arr ):
    """ Documentation: Enter goose to see goosebump graph
    Enter nogoose to see no goosebump graph
    Enter marker to see marker graph
    """
    class_arr = None
    if class_name == 'goose':
        i = 0
        class_arr = raw_prediction_arr[..., i]
        class_list = [1 if prob > 0.8 else 0 for prob in class_arr]
        class_frame = [idx for idx in range(len(class_list)) if class_list[idx] == 1]

    elif class_name == 'nogoose':
        i = 2
        class_arr = raw_prediction_arr[..., i]
        class_list = [1 if prob > 0.5 else 0 for prob in class_arr]

    elif class_name == 'marker':
        i = 1
        class_arr = raw_prediction_arr[..., i]
        class_list = [1 if prob > 0.5 else 0 for prob in class_arr]
        class_frame = [idx for idx in range(len(class_list)) if class_list[idx] == 1]
        list_marker_count = []
        counter = 1
        for i in range(len(class_frame) - 1):
            if i == 0:
                list_marker_count.append(counter)
            if class_frame[i + 1] - class_frame[i] > 5:
                counter = counter + 1
                list_marker_count.append(counter)
            else:
                list_marker_count.append('--')

        df = pd.DataFrame(list(zip(class_frame, list_marker_count)),
                          columns=['Frame', 'Cons. Marker instance'])
        class_frame = df
    else:
        raise ValueError("Fn get_graph: Invalid class name entered. Please check the documentation")

    # plt.plot(class_arr)
    # plt.show()

    return class_frame




if __name__=='__main__':

    file_name = '26_c_al'  ##### Need to changed
    file_path = 'D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\Data_raw_predict_' +file_name+'.txt'
    goose_file_path = 'D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\goose_'+ file_name + '.txt'
    marker_file_path = 'D:\\16 Max Plank\\Dev\\Experiments\\12_Cons_model\\Final_analysis\\marker_'+ file_name + '.txt'
# Read raw prediction data
    raw_prediction_arr = read_raw_prediction(file_path)
    data_to_file = get_graph(class_name= 'goose', raw_prediction_arr = raw_prediction_arr)
    write_to_file(goose_file_path, data_to_file)
    data_to_file = get_graph(class_name= 'marker', raw_prediction_arr = raw_prediction_arr)
    write_to_file(marker_file_path, data_to_file)

    print('end')





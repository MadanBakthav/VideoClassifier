# VideoClassifier
A Classifier model to classify the frames of Video

Dependencies
Opencv
tensorflow > 2.0
tqdm
pickle
numpy 
pandas


How to use :
In main.py, you can add the video location which you would to predict. You can also add the location and name of the output files to be written for a respective video.
Once you run the file, you can monitor the status in the progress bar.

Description of files:
1. main.py - it contains the collections of functions to perform the prediction task
2. data_prep.py - it contains function to prepare the video for running the model prediction
3. prediction_v3.py - it contains functions for importation of model, predicting the frames and storing the results
4. postprocessing.py - it contains functions for performing postprocessing, writting the output files

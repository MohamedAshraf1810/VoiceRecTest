import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time



print("put an action")
print("press 1 To Record Voices 0 To Not Record")
action = input()
if action == '1':
    import getVoice
    import getVoicePath
else:
    print("No Recording Needed ...")

#path to training data
source   = "Build_Set/"   
modelpath = "Testing_Models/"
test_file = "Build_Set_Text.txt"
file_paths = open(test_file,'r')

#path to training data
source   = "Testing_Audio/"   

#path where training speakers will be saved
modelpath = "Trained_Speech_Models/"

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models

models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers  = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
error = 0
total_sample = 0.0

print("Press '1' for checking a single Audio or Press '0' for testing a complete set of audio with Accuracy?")
take=int(input().strip())

if take == 1:

    with open('Trained_Speech_Models/person_other.gmm','rb') as f:
        mypickle = cPickle.load(f)
        print("pickle Loaded" , mypickle)
    
        
    print ("Enter the File name from the sample with .wav notation :")
    path =input().strip()
    print (("Testing Audio : ",path))
    sr,audio = read(source + path)
    vector = extract_features(audio,sr)
    print(vector)
    scores = np.array(mypickle.score(vector))
    print(scores)


    ygmm_pred_class = mypickle.predict_proba(vector)
    plt.scatter(ygmm_pred_class[0:, :], ygmm_pred_class[:,:])
    plt.show()
    # print(ygmm_pred_class)









        

elif take == 0:
    test_file = "Testing_audio_Path.txt"        
    file_paths = open(test_file,'r')
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        total_sample+= 1.0
        path=path.strip()
        print("Testing Audio : ", path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models)) 
        for i in range(len(models)):
            gmm  = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner=np.argmax(log_likelihood)
        print ("\tdetected as - ", speakers[winner])
        checker_name = path.split("_")[0]
        print("CheckerName is ",checker_name)
        if speakers[winner] != checker_name:
            error += 1
        time.sleep(1.0)
        print("-----------------------------------")
    print ("Error : ",error,"Total Samples : ", total_sample)
    accuracy = ((total_sample - error) / total_sample) * 100

    print ("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ",  "%")
    print(accuracy)
    
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from FeatureExtraction import extract_features
import getVoicePath

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")
voiceEncoding=[]
labels = []

source   = "Voice_Samples_Training/"

print("\n\nTraining Started")
dest = "Trained_Speech_Models/"
train_file = "Voice_Samples_Training_Path.txt"
file_paths = open(train_file,'r')


counter=0
count = 1
# Extracting features for each speaker (5 files per speakers)

features = np.asarray(())
for path in file_paths:    
    path = path.strip()
    print ("path is" , path)
    counter+=1

    
    # read the audio
    sr,audio = read(source+path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
        voiceEncoding.append(vector)
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    if count == 5:
        fLabels = path.split("-")[0]
        labels.append(fLabels)
        
        count = 0

        gmmclf = GaussianMixture(n_components=5, covariance_type='diag')
        gmmclf.fit(vector, fLabels) #X_train are mfcc vectors, y_train are labels
        plt.scatter(vector[:, 0], vector[:, 1])
        # plt.show()
        # print(vector)

    count = count + 1   
print(labels)

with open ('Trained_Speech_Models/person_other.gmm','wb') as f:
    pickle.dump(gmmclf,f)
print("trainning Completed")
print("Number Of paths is =",counter)
    


# print("array encoding",voiceEncoding)
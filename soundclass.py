#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.pyplot import specgram
import glob
import os
import pandas as pd

#Defining the functions
def extract_feature(file_name):
																				y, sr = librosa.load(file_name)
																				stft = np.abs(librosa.stft(y))
																				mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T,axis=0)
																				chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
																				mel = np.mean(librosa.feature.zero_crossing_rate(y, sr=sr).T,axis=0)
																				contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
																				tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),
																				    sr=sr).T,axis=0)
																				return mfccs,chroma,mel,contrast,tonnetz
							
    
def parse_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,173)), np.empty(0)
    for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
									
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
												     
            labels = np.append(labels, fn.split('\\')[2].split('-')[3].split('.')[0])
    return np.array(features), np.array(labels, dtype = np.int)

parent_dir = 'audio'
tr_sub_dirs = ["Fold-1","Fold-2","Fold-3","Fold-4"]
ts_sub_dirs = ["Fold-5"]
X_train, y_train = parse_files(parent_dir,tr_sub_dirs)
X_test, y_test = parse_files(parent_dir,ts_sub_dirs)


#K-Nearest Neighbors (K-NN)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#SVM

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, decision_function_shape='ovo')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

#Random FOreset 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) 
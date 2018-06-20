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
	zcr=np.mean(librosa.feature.zero_crossing_rate(y).T,axis=0)
	return zcr
							
    
def parse_files(parent_dir,sub_dirs,file_ext="*.wav"):
     labels = np.empty(0)
     features = np.empty((0,1))
     for sub_dir in sub_dirs:
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
									
            try:
              zcr = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = zcr
            features = np.vstack([features,ext_features])
												     
            labels = np.append(labels, fn.split('\\')[2].split('-')[3].split('.')[0])
     return np.array(features), np.array(labels, dtype = np.int)

parent_dir = 'audio'
tr_sub_dirs = ["Fold-1","Fold-2","Fold-3","Fold-4"]
ts_sub_dirs = ["Fold-5"]
X_train, y_train = parse_files(parent_dir,tr_sub_dirs)
X_test, y_test = parse_files(parent_dir,ts_sub_dirs)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Importing the libraries
import numpy as np
import librosa
import librosa.display
import glob
import os
import json

#Defining the functions
def extract_feature(file_name):
		y, sr = librosa.load(file_name)
		stft = np.abs(librosa.stft(y))
		mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T,axis=0)
		chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
		mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
		contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
		tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),
		    sr=sr).T,axis=0)
		zcr = np.mean(librosa.feature.zero_crossing_rate(y).T,axis=0)
return mfccs,chroma,mel,contrast,tonnetz,zcr
							
    
def parse_files(parent_dir,sub_dirs,file_ext="*.wav"):
	features, labels = np.empty((0,174)), np.empty(0)
	for sub_dir in sub_dirs:
		for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
			print(fn)
			print('\n\n')
			try:
				mfccs, chroma, mel, contrast,tonnetz,zcr = extract_feature(fn)
			except Exception as e:
				print ("Error encountered while parsing file: ", fn)
				continue
			ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz,zcr])  #Horizontally stacking all the features 
			features = np.vstack([features,ext_features])  #Vertically stack features of each sudio one by one

			labels = np.append(labels, fn.split('/')[9].split('-')[3].split('.')[0])
	return np.array(features), np.array(labels, dtype = np.int)





#Input the directory of your audio files
parent_dir = 'audio'
tr_sub_dirs = ["Fold-1","Fold-2","Fold-3"]#Training data direcctory
#tr_sub_dirs = ["Fold-1"]#Training data direcctory
ts_sub_dirs = ["Fold-5"] #test data folder
val_sub_dirs = ["Fold-4"] #validation data
X_feature_train, y_feature_train = parse_files(parent_dir,tr_sub_dirs)
X_feature_test, y_feature_test = parse_files(parent_dir,ts_sub_dirs)
X_feature_val, y_feature_val = parse_files(parent_dir,val_sub_dirs)

#Different training,testing and validation feature datasets extraction		
X_mfcc_train=X_feature_train[:,0:20]
X_chroma_train=X_feature_train[:,20:32]
X_mel_train=X_feature_train[:,32:160]
X_spectral_train=X_feature_train[:,160:175]

X_mfcc_test=X_feature_test[:,0:20]
X_chroma_test=X_feature_test[:,20:32]
X_mel_test=X_feature_test[:,32:160]
X_spectral_test=X_feature_test[:,160:175]

X_mfcc_val=X_feature_val[:,0:20]
X_chroma_val=X_feature_val[:,20:32]
X_mel_val=X_feature_val[:,32:160]
X_spectral_val=X_feature_val[:,160:175]

#-------------------------------svm training-----------------------------
# Feature Scaling of each feature datasets
	
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_mfcc_train = sc.fit_transform(X_mfcc_train)
X_chroma_train = sc.fit_transform(X_chroma_train)
X_mel_train = sc.fit_transform(X_mel_train)
X_spectral_train = sc.fit_transform(X_spectral_train)

X_mfcc_test = sc.fit_transform(X_mfcc_test)
X_chroma_test = sc.fit_transform(X_chroma_test)
X_mel_test = sc.fit_transform(X_mel_test)
X_spectral_test = sc.fit_transform(X_spectral_test)

X_mfcc_val = sc.fit_transform(X_mfcc_val)
X_chroma_val = sc.fit_transform(X_chroma_val)
X_mel_val = sc.fit_transform(X_mel_val)
X_spectral_val = sc.fit_transform(X_spectral_val)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(probability=True)
X_training=[X_mfcc_train,X_chroma_train,X_mel_train,X_spectral_train]
kernelList = ['linear','rbf']


#---------------------------Multiple kernel learning using grid search
def multiplekernel(classifier,X_train,y_train,kernelType):#classifier is our svm model and X_train,y_train represents training datasetand their labels respectively
	#Grid Search
	# Applying Grid Search to find the best model and the best parameters
	# Change parameters argument in accordance with your need
 from sklearn.model_selection import GridSearchCV

 if(kernelType=='linear'):
      		parameters = [{'C': [1,2,4,8,16,32,64,128,256,512,1024,2048], 'kernel': ['linear']}]
      		grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
      		grid_search = grid_search.fit(X_train, y_train)
      		best_accuracy = grid_search.best_score_
      		best_parameters = grid_search.best_params_
 elif(kernelType=='rbf'):
      		parameters = [{'C': [1,2,4,8,16,32,64,128,256,512,1024,2048], 'kernel': ['rbf'],'gamma': [0.01, 0.001, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1,2,5,10]}]
      		grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
      		grid_search = grid_search.fit(X_train, y_train)
      		best_accuracy = grid_search.best_score_
      		best_parameters = grid_search.best_params_
 return best_accuracy,best_parameters


list = []*(len(X_training)*len(kernelList))
for X_train in X_training:
	for kernel in kernelList:
		parameters = multiplekernel(classifier,X_train,y_feature_train,kernel)
		list.append(parameters)



accuracy_dict={'mfcc_ln':list[0],'mfcc_rbf':list[1],'chroma_ln':list[2],'chroma_rbf':list[3],'mel_ln':list[4],'mel_rbf':list[5],'spectral_ln':list[6],'spectral_rbf':list[7]}


#------------------------Training and testing svm again using best parameters obtained from grid
#mfcc 
mfcc_classifier_ln=SVC(C=accuracy_dict['mfcc_ln'][1]['C'],kernel='linear',decision_function_shape='ovo',probability=True,random_state=0)
mfcc_classifier_ln.fit(X_mfcc_train,y_feature_train)
y_pred_ln = mfcc_classifier_ln.predict(X_mfcc_test)

mfcc_classifier_rbf=SVC(C=accuracy_dict['mfcc_rbf'][1]['C'],kernel='rbf',gamma=accuracy_dict['mfcc_rbf'][1]['gamma'],decision_function_shape='ovo',probability=True,random_state=0)
mfcc_classifier_rbf.fit(X_mfcc_train,y_feature_train)
y_pred_rbf = mfcc_classifier_rbf.predict(X_mfcc_test)

test_accuracy=[]*8
from sklearn.metrics import accuracy_score
mfcc_accuracy_ln = accuracy_score(y_feature_test,y_pred_ln)
test_accuracy.append(mfcc_accuracy_ln)

mfcc_accuracy_rbf = accuracy_score(y_feature_test,y_pred_rbf)
test_accuracy.append(mfcc_accuracy_rbf)


#chroma
chroma_classifier_ln=SVC(C=accuracy_dict['chroma_ln'][1]['C'],kernel='linear',decision_function_shape='ovo',probability=True,random_state=0)
chroma_classifier_ln.fit(X_chroma_train,y_feature_train)
y_pred_ln = chroma_classifier_ln.predict(X_chroma_test)

chroma_classifier_rbf=SVC(C=accuracy_dict['chroma_rbf'][1]['C'],kernel='rbf',gamma=accuracy_dict['chroma_rbf'][1]['gamma'],decision_function_shape='ovo',probability=True,random_state=0)
chroma_classifier_rbf.fit(X_chroma_train,y_feature_train)
y_pred_rbf = chroma_classifier_rbf.predict(X_chroma_test)


chroma_accuracy_ln = accuracy_score(y_feature_test,y_pred_ln)
test_accuracy.append(chroma_accuracy_ln)

chroma_accuracy_rbf = accuracy_score(y_feature_test,y_pred_rbf)
test_accuracy.append(chroma_accuracy_rbf)

#------------mel
mel_classifier_ln=SVC(C=accuracy_dict['mel_ln'][1]['C'],kernel='linear',decision_function_shape='ovo',probability=True,random_state=0)
mel_classifier_ln.fit(X_mel_train,y_feature_train)
y_pred_ln = mel_classifier_ln.predict(X_mel_test)

mel_classifier_rbf=SVC(C=accuracy_dict['mel_rbf'][1]['C'],kernel='rbf',gamma=accuracy_dict['mel_rbf'][1]['gamma'],decision_function_shape='ovo',probability=True,random_state=0)
mel_classifier_rbf.fit(X_mel_train,y_feature_train)
y_pred_rbf = mel_classifier_rbf.predict(X_mel_test)

mel_accuracy_ln = accuracy_score(y_feature_test,y_pred_ln)
test_accuracy.append(mel_accuracy_ln)

mel_accuracy_rbf = accuracy_score(y_feature_test,y_pred_rbf)
test_accuracy.append(mel_accuracy_rbf)

#-------spectral
spectral_classifier_ln=SVC(C=accuracy_dict['spectral_ln'][1]['C'],kernel='linear',decision_function_shape='ovo',probability=True,random_state=0)
spectral_classifier_ln.fit(X_spectral_train,y_feature_train)
y_pred_ln = spectral_classifier_ln.predict(X_spectral_test)

spectral_classifier_rbf=SVC(C=accuracy_dict['spectral_rbf'][1]['C'],kernel='rbf',gamma=accuracy_dict['mel_rbf'][1]['gamma'],decision_function_shape='ovo',probability=True,random_state=0)
spectral_classifier_rbf.fit(X_spectral_train,y_feature_train)
y_pred_rbf = spectral_classifier_rbf.predict(X_spectral_test)

spectral_accuracy_ln = accuracy_score(y_feature_test,y_pred_ln)
test_accuracy.append(spectral_accuracy_ln)

spectral_accuracy_rbf = accuracy_score(y_feature_test,y_pred_rbf)
test_accuracy.append(spectral_accuracy_rbf)


Accuracy_dictionary={'mfcc_ln':test_accuracy[0],'mfcc_rbf':test_accuracy[1],
                     'chroma_ln':test_accuracy[2],'chroma_rbf':test_accuracy[3],
                     'mel_ln':test_accuracy[4],'mel_rbf':test_accuracy[5],
                     'spectral_ln':test_accuracy[6],'spectral_rbf':test_accuracy[7]}

#------------------------ENSEMBLE LEARNING-----------------------
from sklearn.svm import SVR
classifiers = [mfcc_classifier_ln,mfcc_classifier_rbf,chroma_classifier_ln,chroma_classifier_rbf,
               mel_classifier_ln,mel_classifier_rbf,spectral_classifier_ln,spectral_classifier_rbf]
X_reg=[X_mfcc_val,X_mfcc_val,X_chroma_val,X_chroma_val,
							X_mel_val,X_mel_val,X_spectral_val,X_spectral_val]
regression = []
i=0
for classifier in classifiers:
                y=[]
                y_pred=classifier.predict(X_reg[i])
for j in range(len(y_pred)):
	if(y_pred[j]==y_feature_val[j]):
			 y.append(1)
	elif(y_pred[j]!=y_feature_val[j]):
		         y.append(0)
					  # Fitting SVR to the dataset
                regressor = SVR(kernel = 'rbf')
                regressor.fit(X_reg[i],y)
                regression.append(regressor)
                i=i+1
X_test=[X_mfcc_test,X_mfcc_test,X_chroma_test,X_chroma_test,
		X_mel_test,X_mel_test,X_spectral_test,X_spectral_test]



final = np.zeros((400,50),dtype = np.float32)#change shape according to your test dataset

for i in range(8):
	yreg=regression[i].predict(X_test[i])                
	ycl=classifiers[i].predict_proba(X_test[i])
	stk = []
	for k in range(len(yreg)):
				y_new=(ycl[k]*yreg[k])
				stk.append(y_new)

	arr = np.array(stk,dtype = np.float32)
	final+=arr

y_pred = []

for i in range(400):
	l =[]
	for j in range(50):
		l.append(arr[i][j])

	y_pred.append(l.index(max(l)))

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_feature_test)    
with open('Accuracy.txt','w')as file:#writing accuracy to text file
      file.write(json.dumps(Accuracy_dictionary))
print('Accuracy of ensemble classifier is :',accuracy)
print('Accuracy of different classifiers with best parameters obtained from grid search:',Accuracy_dictionary)

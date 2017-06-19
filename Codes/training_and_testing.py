import os
import json
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

training_set  = np.load('./../Standalone/training_set.npy')
training_set = shuffle(training_set)
training_data = training_set[:,0:25]
training_label = training_set[:,25]

# Finding Best Parameters : Grid Search

# c_range = np.logspace(start=-20, stop=20, num=100, base=2)
# class_weight = [None,'balanced']
# param_grid = dict(C=c_range, class_weight=class_weight)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv)
# grid.fit(training_data, training_label)
# print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

clf = LinearSVC(C=3.53, class_weight='balanced')
clf.fit(training_data, training_label)

filename = './../Standalone/kmeans_model.sav'
kmeans = pickle.load(open(filename, 'rb'))
print "Kmeans Model loaded"

'''
input parameters for testing data
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'atrium_3'

#-----------------------------------------------

from features import create_feature_vector

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy')
spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
prediction_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_pred.npy')

start_frame_no = 1
end_frame_no = no_frames - 1
testing_label = []
predicted_label = []

for frame in range(start_frame_no, end_frame_no):
	print frame
	curr_frame_unique_foreground_values = np.delete(np.unique(foreground_arr[frame]),0)

	for val in curr_frame_unique_foreground_values: # Iterates over all foreground regions

		codeword_histogram = np.zeros(25, dtype=np.int32)
		X,Y=np.where(foreground_arr[frame]==val) # Locations of a particular foreground region
		for i in range(len(X)):
			feature_vector = create_feature_vector(no_frames, prediction_data, \
				spatial_mv_data, frame, X[i], Y[i])

			index = kmeans.predict(np.array(feature_vector).reshape(1,-1))[0]
			codeword_histogram[index]+=1

		normalized_histogram = codeword_histogram/float(sum(codeword_histogram))
		prediction = clf.predict(np.array(normalized_histogram).reshape(1,-1))[0]

		predicted_label+=[prediction]
		
		if video_type=='Vehicle_dataset':
			testing_label+=[1]
		elif video_type=='Pedestrian_dataset':
			testing_label+=[0]

try:
	old_testing_label = np.load('./../Standalone/testing_label.npy')
	old_predicted_label = np.load('./../Standalone/predicted_label.npy')

	testing_label = np.concatenate((old_testing_label, np.array(testing_label)))
	predicted_label = np.concatenate((old_predicted_label, np.array(predicted_label)))

	np.save('./../Standalone/testing_label.npy', testing_label)
	np.save('./../Standalone/predicted_label.npy', predicted_label)

except:
	np.save('./../Standalone/testing_label.npy', np.array(testing_label))
	np.save('./../Standalone/predicted_label.npy', np.array(predicted_label))
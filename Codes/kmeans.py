'''
Runs kmeans on the triaining features and creates training data
'''
import os
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans

feature_arr = np.load('./../Standalone/features.npy')

kmeans = KMeans(n_clusters=25, max_iter=1000).fit(feature_arr) # have to check

filename = './../Standalone/kmeans_model.sav'
pickle.dump(kmeans, open(filename, 'wb'))

# kmeans = pickle.load(open(filename, 'rb'))
# print "Model loaded"

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Vehicle_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'sherbrooke_4'

#-----------------------------------------------

foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy')
spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
prediction_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_pred.npy')

from features import create_feature_vector

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

no_frames=  video_data[video_type][video_name]["no_frames"]
start_frame_no = 1
end_frame_no = no_frames - 1
codeword_histogram = np.zeros(25, dtype=np.int32)
training_set = []
training_label = []

for frame in range(start_frame_no, end_frame_no): #HAVE TO ASK RANGE
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
		training_set+=[normalized_histogram]
		if video_type=='Vehicle_dataset':
			training_label+=[1]
		elif video_type=='Pedestrian_dataset':
			training_label+=[0]

training_label = np.array(training_label)

try:
	old_training_set = np.load('./../Standalone/training_set.npy')

	training_label = training_label.reshape(-1, 1)
	training_set = np.concatenate((training_set, training_label), axis=1)
	
	training_set = np.concatenate((old_training_set, np.array(training_set)))
	np.save('./../Standalone/training_set.npy', training_set)

except:
	training_label = training_label.reshape(-1, 1)
	training_set = np.concatenate((training_set, training_label), axis=1)
	np.save('./../Standalone/training_set.npy', np.array(training_set))
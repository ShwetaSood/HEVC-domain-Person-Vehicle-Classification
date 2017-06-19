'''
Results in mv frame with sizes: 1,2,4
'''

import os
import cv2
import sys
import json
import math
import time
import numpy as np
from ast import literal_eval

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'pedestrians'

#-----------------------------------------------

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

filename = './../'+video_type+'/'+video_name+'/decoder_files'
files = ['decoder_cupu.txt','decoder_pred.txt','decoder_mv.txt']
input_image_path = './../'+video_type+'/'+video_name+'/input'
input_images =  np.sort(os.listdir(input_image_path))

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

no_ctu_width = int(math.ceil(video_width/64.0))
no_ctu_height = int(math.ceil(video_height/64.0))

copy_spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_interpolated.npy')
spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_interpolated.npy')
[frames ,height, width] = spatial_mv_data.shape

start_frame_no = 4
end_frame_no = no_frames - 4

start_time = time.time()

for frame in range(start_frame_no, end_frame_no):
	print frame
	for i in range(len(spatial_mv_data[frame])):
		for j in range(len(spatial_mv_data[frame,0,:])):

			if len(literal_eval(spatial_mv_data[frame, i, j]))==4 or\
			   len(literal_eval(spatial_mv_data[frame, i, j]))==2:

				total_temporal_x = literal_eval(spatial_mv_data[frame, i, j])[-2]
				total_temporal_y = literal_eval(spatial_mv_data[frame, i, j])[-1]

			for temporal_val in range(4,0,-1):
				if len(literal_eval(spatial_mv_data[frame-temporal_val, i, j]))==4 or\
				   len(literal_eval(spatial_mv_data[frame-temporal_val, i, j]))==2:

					total_temporal_x+=literal_eval(spatial_mv_data[frame-temporal_val, i, j])[-2]
					total_temporal_y+=literal_eval(spatial_mv_data[frame-temporal_val, i, j])[-1]

			for temporal_val in range(1,5):
				if len(literal_eval(spatial_mv_data[frame+temporal_val, i, j]))==4 or\
				   len(literal_eval(spatial_mv_data[frame+temporal_val, i, j]))==2:

					total_temporal_x+=literal_eval(spatial_mv_data[frame+temporal_val, i, j])[-2]
					total_temporal_y+=literal_eval(spatial_mv_data[frame+temporal_val, i, j])[-1]

			if len(literal_eval(spatial_mv_data[frame, i, j]))==1:
				if (float(math.floor(total_temporal_x/9.0))==0.0) and \
				(float(math.floor(total_temporal_y/9.0))==0.0):
				
					copy_spatial_mv_data[frame, i, j] = str([0])
				else:
					copy_spatial_mv_data[frame, i, j] = str([int(math.floor(total_temporal_x/9.0)), \
												int(math.floor(total_temporal_y/9.0))])

			elif len(literal_eval(spatial_mv_data[frame, i, j]))==4:
				mv_vector = literal_eval(spatial_mv_data[frame, i, j])
				mv_vector[2] = int(math.floor(total_temporal_x/9.0))
				mv_vector[3] = int(math.floor(total_temporal_y/9.0))
				copy_spatial_mv_data[frame, i, j] = str(mv_vector)

end_time = time.time()
print "Time taken: ", str(int(end_time-start_time)), " seconds"
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_filtering.npy',copy_spatial_mv_data)

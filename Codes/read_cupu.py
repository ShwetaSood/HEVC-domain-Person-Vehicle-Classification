'''
reading and writing in 4x4 blocks : spatial translation of decoder files
'''
import os
import cv2
import sys
import math
import json
import numpy as np
import pandas as pd

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Vehicle_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'sherbrooke_4'

#-----------------------------------------------

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

filename = './../'+video_type+'/'+video_name+'/decoder_files'
files = ['decoder_cupu.txt','decoder_pred.txt','decoder_mv.txt']
input_image_path = './../'+video_type+'/'+video_name+'/input/'
input_images =  np.sort(os.listdir(input_image_path))

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

no_ctu_width = int(math.ceil(video_width/64.0)) # CTU Size = 64 (Constant)
no_ctu_height = int(math.ceil(video_height/64.0))
pos_pred = 0
pos_mv = 0
pic_pred = np.zeros(shape=[no_frames,video_height/4,video_width/4], dtype='int64')
pic_depth = np.zeros(shape=[no_frames,video_height/4,video_width/4], dtype='int64')
pic_mv = np.chararray([no_frames,video_height/4,video_width/4], itemsize=100)

with open(filename+'/decoder_cupu.txt') as f1:
	data = f1.readlines()
with open(filename+'/decoder_pred.txt') as f2:
	pred_data = f2.readlines()
with open(filename+'/decoder_mv.txt') as f3:
	mv_data = f3.readlines()

def decode_cupu(pos_cupu, cupu_list, pred_list, mv_list, start_w, start_h, \
	cu_width, cu_height, cu_depth):
	'''
	pos_cupu: position in cupu_list
	'''
	global pos_mv
	global pos_pred
	global frame_no
	count = 0
	i = pos_cupu
	while i<len(cupu_list):
		if count==0:
			temp_start_w = start_w
			temp_start_h = start_h
		elif count==1:
			temp_start_w = start_w + cu_width
			temp_start_h = start_h
		elif count==2:
			temp_start_w = start_w
			temp_start_h = start_h + cu_height
		elif count==3:
			temp_start_w = start_w + cu_width
			temp_start_h = start_h + cu_height
		elif count==4:
			return i-1

		if cupu_list[i]=='99':
			i = decode_cupu(i+1, cupu_list, pred_list, mv_list, temp_start_w, temp_start_h, \
				cu_width/2, cu_height/2, cu_depth+1)
			count+=1
		elif cupu_list[i]=='0':
			pic_depth[frame_no, temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = cu_depth#pred_list[pos_pred]

			pic_pred[frame_no, temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = pred_list[pos_pred]
			pos_pred += 1

			curr_mv_list = []
			if mv_list[pos_mv]==1:
				curr_mv_list = mv_list[pos_mv:pos_mv+4] # take next 4 elements
				pos_mv += 4
			elif mv_list[pos_mv]==0:
				curr_mv_list = [0] # contains no MV
				pos_mv += 1
			elif mv_list[pos_mv]==2:
				print "Two reference frames, haven't handled this"
				raise SystemExit

			pic_mv[frame_no,temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = str(curr_mv_list)

			count+=1

			# cv2.line(img,(temp_start_w, temp_start_h),(cu_width+temp_start_w, temp_start_h),(255,255,255),1)
			# cv2.line(img,(temp_start_w,temp_start_h),(temp_start_w,cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,(cu_width+temp_start_w, temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,(temp_start_w,cu_height+temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			# cv2.imshow('image',img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

		elif cupu_list[i]=='3': # Entering a PU

			pic_depth[frame_no,temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = cu_depth+1 # All 4 PUs have to be at the same depth

			assert pred_list[pos_pred]== pred_list[pos_pred+1]== \
			pred_list[pos_pred+2]== pred_list[pos_pred+3]
			
			pic_pred[frame_no,temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = pred_list[pos_pred]
			pos_pred += 4

			iterator = 0
			while iterator<4:
				curr_mv_list = []
				if mv_list[pos_mv]==1:
					curr_mv_list = mv_list[pos_mv:pos_mv+4] # take next 4 elements
					pos_mv += 4
				elif mv_list[pos_mv]==0:
					curr_mv_list = [0] # contains no MV
					pos_mv += 1
				elif mv_list[pos_mv]==2:
					print "Two reference frames, haven't handled this"
					raise SystemExit

				if iterator>0:
					assert(old_str==str(curr_mv_list))
				old_str = str(curr_mv_list)

				iterator+=1

			pic_mv[frame_no,temp_start_h/4:(temp_start_h+cu_height)/4,\
			temp_start_w/4:(temp_start_w+cu_width)/4] = str(curr_mv_list)

			count+=1
			# cv2.line(img,(temp_start_w, temp_start_h),(cu_width+temp_start_w, temp_start_h),(255,255,255),1)
			# cv2.line(img,(temp_start_w,temp_start_h),(temp_start_w,cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,(cu_width+temp_start_w, temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,(temp_start_w,cu_height+temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,((temp_start_w+cu_width+temp_start_w)/2, temp_start_h),((temp_start_w+cu_width+temp_start_w)/2,cu_height+temp_start_h),(255,255,255),1)
			# cv2.line(img,(temp_start_w,(temp_start_h+cu_height+temp_start_h)/2),(temp_start_w+cu_width,(temp_start_h+cu_height+temp_start_h)/2), (255,255,255),1)
			# cv2.imshow('image',img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		elif cupu_list[i]=='15':
			count+=1
		i+=1

	return i

for index in range(len(data)):

	cupu_info, cupu_struct= data[index].split('> ')
	pred_list = pred_data[index].split('> ')[1].strip('\n\r ').split(' ')
	pred_list = [int(el) for el in pred_list]
	mv_list = mv_data[index].split('> ')[1].strip('\n\r ').split(' ')
	mv_list = [int(el) for el in mv_list]
	frame_no, ctu_no= cupu_info.strip('<').split(',')
	frame_no = int(frame_no)

	if frame_no==0 and int(ctu_no)==0: # reading the image for first time
		print frame_no
		curr_frame_no = 0
		img = cv2.imread(input_image_path+input_images[curr_frame_no], cv2.IMREAD_COLOR)

	if frame_no!=curr_frame_no:
		print frame_no
		curr_frame_no = frame_no
		img = cv2.imread(input_image_path+input_images[curr_frame_no], cv2.IMREAD_COLOR)
	pos_pred = 0
	pos_mv = 0
	cupu_list = cupu_struct.strip('\n\r ').split(' ')
	decode_cupu(0, cupu_list, pred_list, mv_list, \
		64*(int(ctu_no)%no_ctu_width), 64*(int(ctu_no)/no_ctu_width), 64, 64, 0)

np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_pred.npy',pic_pred)
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_depth.npy',pic_depth)
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_mv.npy',pic_mv)

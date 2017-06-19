'''
only for reading from decoder files
'''

import os
import cv2
import sys
import math
import json
import numpy as np
import pandas as pd

filename = 'hall_monitor/decoder_files'
files = ['decoder_cupu.txt','decoder_pred.txt','decoder_mv.txt']
video_width = 352
video_height = 288
no_frames=  1
no_ctu_width = int(math.ceil(video_width/64.0))
no_ctu_height = int(math.ceil(video_height/64.0))

with open(filename+'/decoder_cupu.txt') as f1:
	data = f1.readlines()
with open(filename+'/decoder_pred.txt') as f2:
	pred_data = f2.readlines()
with open(filename+'/decoder_mv.txt') as f3:
	mv_data = f3.readlines()

def decode_cupu(pos_cupu, cupu_list, start_w, start_h, \
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
			i = decode_cupu(i+1, cupu_list, temp_start_w, temp_start_h, \
				cu_width/2, cu_height/2, cu_depth+1)
			count+=1
		elif cupu_list[i]=='0':
			count+=1

			# print cu_height
			# print cu_width
			# print cu_depth
			# print temp_start_w
			# print temp_start_h
			cv2.line(img,(temp_start_w, temp_start_h),(cu_width+temp_start_w, temp_start_h),(255,255,255),1)
			cv2.line(img,(temp_start_w,temp_start_h),(temp_start_w,cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,(cu_width+temp_start_w, temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,(temp_start_w,cu_height+temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			cv2.imshow('image',img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		elif cupu_list[i]=='3': # Entering a PU

			count+=1
			# print cu_height
			# print cu_width
			# print cu_depth
			# print temp_start_w
			# print temp_start_h
			cv2.line(img,(temp_start_w, temp_start_h),(cu_width+temp_start_w, temp_start_h),(255,255,255),1)
			cv2.line(img,(temp_start_w,temp_start_h),(temp_start_w,cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,(cu_width+temp_start_w, temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,(temp_start_w,cu_height+temp_start_h),(cu_width+temp_start_w, cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,((temp_start_w+cu_width+temp_start_w)/2, temp_start_h),((temp_start_w+cu_width+temp_start_w)/2,cu_height+temp_start_h),(255,255,255),1)
			cv2.line(img,(temp_start_w,(temp_start_h+cu_height+temp_start_h)/2),(temp_start_w+cu_width,(temp_start_h+cu_height+temp_start_h)/2), (255,255,255),1)
			cv2.imshow('image',img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		elif cupu_list[i]=='15':
			count+=1
		i+=1

	return i

for index in range(len(data)):
	cupu_info, cupu_struct= data[index].split('> ')

	frame_no, ctu_no= cupu_info.strip('<').split(',')
	print frame_no
	frame_no = int(frame_no)
	name = '%06d' %(frame_no+1)
	img = cv2.imread('./hall_monitor/in'+name+'.jpg',cv2.IMREAD_COLOR)
	pos_pred = 0
	pos_mv = 0
	cupu_list = cupu_struct.split(' ')
	decode_cupu(0, cupu_list, 64*(int(ctu_no)%no_ctu_width), 64*(int(ctu_no)/no_ctu_width), 64, 64, 0)
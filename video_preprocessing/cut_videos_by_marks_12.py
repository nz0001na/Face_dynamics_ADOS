"""
    This program cut video into several scenarios by time marks
"""

import cv2
import os
import csv

ro = '/home/na/3_ASD_micro_expression/2_data_ADOS/1_TN/'
src_path = ro + '1_raw_videos/'

# ES_190418_Female, EK_190514_Female, MK_190503_Male, MP_190509_Female
name = 'MP_190509_Female'
dst_path = ro + '2_partition_videos/' + name + '/'

# time marks
# # ES_190418_Female
# info = ['0:02:10', '0:06:54', '0:08:07', '0:10:35', '0:13:05', '0:18:55',
#         '0:23:42', '0:26:16', '0:28:39', '0:35:24', '0:41:30', '0:50:17', '0:51:46',
#         '0:54:57',	'1:00:57']
# # EK_190514_Female
# info = ['0:05:48', '0:11:53', '0:14:00', '0:17:46',	'0:28:09', '0:31:09',
#         '0:46:13',	'0:48:30', '50:58:00', '0:57:16', '1:03:59', '1:10:00',
#         '1:17:49', '1:26:11', '1:31:56']
# MK_190503_Male
# info = ['0:03:41', '0:08:03', '0:10:20', '0:14:55', '0:22:00', '0:30:40',
#         '0:38:17', '0:48:10', '0:42:41', '0:51:06', '1:00:33', '1:09:12',
#         '1:11:12', '1:12:41', '1:20:49']
# MP_190509_Female
info = ['0:02:40', '0:09:06', '0:15:12', '0:18:15', '0:23:32', '0:34:14',
        '0:39:27', '0:41:53', '0:44:43', '0:51:17', '0:56:02', '1:08:04',
        '1:09:53', '1:10:58', '1:16:18']

time_mark = []
for time_m in info:
    time_ms = time_m.split(':')
    time_mm = int(time_ms[0]) * 60 * 60 + int(time_ms[1]) * 60 + int(time_ms[2])
    time_mark.append(time_mm)

v_file = src_path + name + '.MXF'
print(name)


videoCapture = cv2.VideoCapture(v_file)
if (videoCapture.isOpened()):
    print(name + ' Open')
else:
    print('Fail to open ' + name)

# get frame rate, size, codec
fps = videoCapture.get(cv2.CAP_PROP_FPS)
print('fps=' + str(fps))
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
# videoCapture.get(cv2.CAP_PROP_)

# clip = VideoFileClip(filename)
# file_time = self.timeConvert(clip.duration)

number_sce = len(time_mark)
n = 0
start_sec = 1   # which scenario does it start?
for n in range(number_sce):
    scenario_id = start_sec + n
    print(scenario_id)
    dst_name = dst_path + name + '_' + str(scenario_id) + '.avi'

    directory = os.path.dirname(dst_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    videoWriter = cv2.VideoWriter(dst_name, fourcc, fps, size)
    if n == 0:
        # scenario_time = time_mark[n]
        start_frame = 0
        end_frame = int(round(time_mark[n] * fps))
    else:
        # scenario_time = time_mark[n] - time_mark[n - 1]
        start_frame = int(round(time_mark[n-1] * fps))
        end_frame = int(round(time_mark[n] * fps))

    i = 1
    frame_count = end_frame - start_frame
    for i in range(start_frame, end_frame):
        # print '%d  %d/%d' % (n, i, frame_count)
        success, frame = videoCapture.read()
        if success:
            videoWriter.write(frame)

videoCapture.release()
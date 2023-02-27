import os
import csv
import cv2

ro = '/home/na/3_ASD_micro_expression/2_data_ADOS/1_TN/'
src_path = ro + '3_frames_raw/'
dst_path = ro + '4_cropped_500x500/'

# ES_190418_Female	430	395	930	895
# MK_190503_Male	486	377	986	877
# MP_190509_Female	345	300	845	800
# EK_190514_Female  174	289	674	789
bbox = [[530, 395, 1030, 895],
        [486, 377, 986, 877],
        [345, 300, 845, 800],
        [174, 289, 674, 789]
        ]

llist = ['ES_190418_Female', 'MK_190503_Male', 'MP_190509_Female', 'EK_190514_Female']

for i in range(len(llist)):
    sub = llist[i]
    if sub not in ['EK_190514_Female']: continue
    box = bbox[i]
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    sces = os.listdir(src_path + sub + '/')
    for sce in sces:
        if sce in ['EK_190514_Female_5']: continue
        print(sub + '/' + sce)

        dst_fold = dst_path + sub + '/' + sce
        if os.path.exists(dst_fold) is False:
            os.makedirs(dst_fold)

        imgs = os.listdir(src_path + sub + '/' + sce)
        for im in imgs:
            img = cv2.imread(src_path + sub + '/' + sce + '/' + im)

            crop_img = img[y1:y2, x1:x2]
            # resized = cv2.resize(crop_img, (270, 270), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_fold + '/' + im, crop_img)

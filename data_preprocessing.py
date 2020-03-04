import numpy as np
import pandas as pd
import csv
from utils import read_db, observe
# source & target view number for differet dataset
# 1:Classroom 2:Museum 3:Hijack
# source view (sv)
list_sv1 = [0,10,11,12,13,7,8]
list_sv2 = [0,1,11,12,13,17,4]
list_sv3 = [1,2,3,4,5,8,9]
# target view (tv)
list_tv1 = [1,2,3,4,5,6,9,14]
list_tv2 = [2,3,4,5,6,7,8,9,10,14,15,16,18,19,20,21,22,23]
list_tv3 = [0,6,7]
# frame oreder
list_f1 = [100,20,40,60,80]
list_f2 = [100,150,200,250,50]
list_f3 = [100,150,200,250,50]
# load data
is_train = True
test_folder = 'obj_nsv'
read_db_obj = read_db(is_train = is_train, test_folder= test_folder)
# 
# Funtion
def get_para(np_cp_obj,IDofdataset):
    if(IDofdataset==1):
        for i in range(len(np_cp_obj)):
            if(np_cp_obj[i][0]=='ClassroomVideo'):
                para = np_cp_obj[i][1].to_numpy()
                para = np.delete(para, [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],axis=1)
                para = np.delete(para,[0,1],axis=0)
                return para
    elif(IDofdataset==2):
        for i in range(len(np_cp_obj)):
            if(np_cp_obj[i][0]=='TechnicolorMuseum'):
                para = np_cp_obj[i][1].to_numpy()
                para = np.delete(para, [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],axis=1)
                para = np.delete(para,[0,1],axis=0)
                return para
    elif(IDofdataset==3):
        for i in range(len(np_cp_obj)):
            if(np_cp_obj[i][0]=='TechnicolorHijack'):
                para = np_cp_obj[i][1].to_numpy()
                para = np.delete(para, [0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],axis=1)
                para = np.delete(para,[0,1],axis=0)
                return para
def get_para_diff(para,list_sv,list_tv):
    para_p=[]
    para_o=[]
    
    for i in range(len(list_tv)):
        temp1=[]
        for j in range(len(list_sv)):
            temp2=[]
            for k in range(3):
                pos = para[list_tv[i]][k]-para[list_sv[j]][k]
                temp2.append(pos)
            temp1.append(temp2)
        para_p.append(temp1)
    
    for i in range(len(list_tv)):
        temp1=[]
        for j in range(len(list_sv)):
            temp2=[]
            for k in range(3):
                ori = para[list_tv[i]][3+k]-para[list_sv[j]][3+k]
                temp2.append(ori)
            temp1.append(temp2)
        para_o.append(temp1)

    return para_p,para_o
def get_sv(np_sv_obj,IDofdataset):
    if (IDofdataset==1):
        for i in range(len(np_sv_obj)):
            if(np_sv_obj[i][0]=='ClassroomVideo'):
                sv_imgs = sv_array[i][1]['imgs']
                sv_depth = sv_array[i][1]['depth']
                return sv_imgs,sv_depth
    if (IDofdataset==2):
        for i in range(len(np_sv_obj)):
            if(np_sv_obj[i][0]=='TechnicolorMuseum'):
                sv_imgs = sv_array[i][1]['imgs']
                sv_depth = sv_array[i][1]['depth']
                return sv_imgs,sv_depth
    if (IDofdataset==3):
        for i in range(len(np_sv_obj)):
            if(np_sv_obj[i][0]=='TechnicolorHijack'):
                sv_imgs = sv_array[i][1]['imgs']
                sv_depth = sv_array[i][1]['depth']
                return sv_imgs,sv_depth
def gray(imgs):
    for i in range(len(imgs)):
        for j in range(len(imgs[0])):
            avg = sum(imgs[i][j])/len(imgs[i][j])
            imgs[i][j] = avg/255
def map_gen_P(para_num):
    para_map=[]
    for i in range(256):
        temp = []
        for j in range(256):
            temp.append(para_num*10)
        para_map.append(temp)
    return para_map
def map_gen_O(para_num):
    para_map=[]
    for i in range(256):
        temp = []
        for j in range(256):
            temp.append(para_num/180)
        para_map.append(temp)
    return para_map
# process data
# print(read_db_obj.sv)
print("Input Source view...")
sv_array = np.array(list(read_db_obj.sv.items()))
sv_imgs1,sv_depth1 = get_sv(sv_array,1)
sv_imgs2,sv_depth2 = get_sv(sv_array,2)
sv_imgs3,sv_depth3 = get_sv(sv_array,3)
sv_imgs1 = sv_imgs1.tolist()
sv_imgs2 = sv_imgs2.tolist()
sv_imgs3 = sv_imgs3.tolist()
sv_depth1 = sv_depth1.tolist()
sv_depth2 = sv_depth2.tolist()
sv_depth3 = sv_depth3.tolist()
for i in range(7):
    for j in range(5):
        gray(sv_imgs1[i][j])
        gray(sv_imgs2[i][j])
        gray(sv_imgs3[i][j])
        gray(sv_depth1[i][j])
        gray(sv_depth2[i][j])
        gray(sv_depth3[i][j])

# print(sv_imgs1)
# parameter
print("Input Parameter")
cp_array = np.array(list(read_db_obj.cp.items()))

para1 = get_para(cp_array,1)
para2 = get_para(cp_array,2)
para3 = get_para(cp_array,3)

para_p1,para_o1 = get_para_diff(para1,list_sv1,list_tv1)
para_p2,para_o2 = get_para_diff(para2,list_sv2,list_tv2)
para_p3,para_o3 = get_para_diff(para3,list_sv3,list_tv3)

print("Produce Training Data")
Train_qurey = []
Train_data = []
# texture,depth,position,orientaiton
Train_label = []
# p1,p2,p3
with open('Testing_query_NSV.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        Train_qurey.append(row)

for i in range(len(Train_qurey)):
    example = []
    # texture = []
    # depth = []
    # position_map = []
    # orientation_map = []
    if(Train_qurey[i][0]=='ClassroomVideo'):
        frame=0
        target=0
        for j in range(len(list_f1)):
            if (int(Train_qurey[i][1])==list_f1[j]):
                frame = j
        for j in range(len(list_tv1)):
            if(int(Train_qurey[i][2])==list_tv1[j]):
                target = j
        for j in range(len(list_sv1)):
            example.append(sv_imgs1[j][frame])
            example.append(sv_depth1[j][frame])
            for k in range(3):
                example.append(map_gen_P(para_p1[target][j][k]))
            for k in range(3):
                example.append(map_gen_O(para_o1[target][j][k]))
    if(Train_qurey[i][0]=='TechnicolorMuseum'):
        frame=0
        target=0
        for j in range(len(list_f2)):
            if (int(Train_qurey[i][1])==list_f2[j]):
                frame = j
        for j in range(len(list_tv2)):
            if(int(Train_qurey[i][2])==list_tv2[j]):
                target = j
        for j in range(len(list_sv2)):
            example.append(sv_imgs2[j][frame])
            example.append(sv_depth2[j][frame])
            for k in range(3):
                example.append(map_gen_P(para_p2[target][j][k]))
            for k in range(3):
                example.append(map_gen_O(para_o2[target][j][k]))
    if(Train_qurey[i][0]=='TechnicolorHijack'):
        frame=0
        target=0
        for j in range(len(list_f3)):
            if (int(Train_qurey[i][1])==list_f3[j]):
                frame = j
        for j in range(len(list_tv3)):
            if(int(Train_qurey[i][2])==list_tv3[j]):
                target = j
        for j in range(len(list_sv3)):
            example.append(sv_imgs3[j][frame])
            example.append(sv_depth3[j][frame])
            for k in range(3):
                example.append(map_gen_P(para_p3[target][j][k]))
            for k in range(3):
                example.append(map_gen_O(para_o3[target][j][k]))
    # example.append(texture)
    # example.append(depth)
    # example.append(position_map)
    # example.append(orientation_map)
    Train_data.append(example)

for i in range(len(Train_qurey)):
    temp=[]
    p1 = int(Train_qurey[i][3])/7
    p2 = (int(Train_qurey[i][4])-int(Train_qurey[i][3]))/7
    p3 = (int(Train_qurey[i][5])-int(Train_qurey[i][4]))/7
    temp.append(p1)
    temp.append(p2)
    temp.append(p3)
    Train_label.append(temp)

T_data = np.asarray(Train_data)
T_label = np.asarray(Train_label)
np.save('datasets/Testing_data_NSV.npy', T_data)
np.save('datasets/Testing_label_NSV.npy',T_label)
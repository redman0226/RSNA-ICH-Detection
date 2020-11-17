from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
import numpy as np
import cv2
import os
import glob
import glob
from os.path import isfile, join
import random

random.seed(42)
inputfolder = 'TrainingData'
imagefolder = 'TrainingData_png'
if not os.path.exists(imagefolder):
    os.mkdir(imagefolder)
seqs = os.listdir(inputfolder)
print(seqs)
fold1, fold2, fold3, fold4, fold5 = [], [],[],[],[],
for seq in seqs:
    path = os.path.join(imagefolder, seq, '*.png')
    #print(path)
    onlyfiles = glob.glob(path)
    random.shuffle(onlyfiles)
    fold1 = fold1+onlyfiles[:200]
    fold2 = fold2+onlyfiles[200:400]
    fold3 = fold3+onlyfiles[400:600]
    fold4 = fold4+onlyfiles[600:800]
    fold5 = fold5+onlyfiles[800:]

for i in range(5):
    index = [fold1, fold2, fold3, fold4, fold5]
    with open('val_'+str(i+1)+'.txt', 'w') as f:
        foldname = index[i]
        for e in foldname:
            f.writelines(e)
            f.writelines(' ')
            f.writelines('\n')
        del index[i]
    print(len(index))
    with open('train_'+str(i+1)+'.txt', 'w') as f:
        #foldname = index[i]
        for fold in index:
            for e in fold:
                f.writelines(e)
                f.writelines(' ')
                f.writelines('\n')
#ds = dcmread('TrainingData\\epidural\\ID_000edbf38.dcm')
#windowed = apply_voi_lut(ds.pixel_array, ds)
#ds = pydicom.read_file('TrainingData\\epidural\\ID_000edbf38.dcm') # read dicom image
#img = ds.pixel_array
#img_normalized = (img-np.min(img))/(np.max(img) - np.min(img))
#print(img_normalized)
#cv2.imwrite(os.path.join('test.png'), img_normalized*255)
'''
for seq in seqs:
    folder = os.path.join(inputfolder, seq)
    outfolder = os.path.join(imagefolder, seq)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for dcmfile in glob.glob(os.path.join(folder, '*.dcm')):
        imname = dcmfile.split('\\')[-1][:-4]
        ds = pydicom.read_file(dcmfile) # read dicom image
        img = ds.pixel_array # get image array
        img_normalized = (img-np.min(img))/(np.max(img) - np.min(img))
        print(os.path.join(outfolder, imname+'.png'))
        cv2.imwrite(os.path.join(outfolder, imname+'.png'), img_normalized*255)
        # Add code for rescaling to 8-bit...
'''

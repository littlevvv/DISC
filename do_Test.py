from predict import MLpridict
import cv2, os, time, csv
from glob import glob
import shutil
import numpy as np

print('ini')
ml = MLpridict('/media/r300/1T/A30335/Disc/past/discmodel/ITRI_CMS_R100_unet_residual_vgg16.hdf5',
               '/media/r300/1T/A30335/Disc/past/grademodel/ITRI_CMSR100_DenseNet121_50_128_loss.hdf5',
               '/media/r300/1T/A30335/Disc/past/vertebraModel/Sacral_R100_unet_vgg16.hdf5',
               '/media/r300/1T/A30335/Disc/past/hiz_R100_unet_vgg16.hdf5')
print('ok')
imgPath = '/media/r300/1T/A30335/Disc/1 (3).png'
folder = '/media/r300/1T/A30335/Disc/testresult'
Evaluation, mean, std, w, h,grade,area,ccx,ccy,ccw,cch = ml.modelPredict(imgPath, folder)
# print(Evaluation, mean, std, w, h ,grade,area,ccx,ccy,ccw,cch)


print(grade)
                
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2, os
from keras import backend as K
import tensorflow as tf
# from PyQt5 import QtCore, QtGui, QtWidgets
import traceback, sys
from datetime import datetime
def swish(x):
    return (K.sigmoid(x) * x)
        
class MLpridict:
    def __init__(self, modelName,gradeModelName, hivdModelName, hizModelName):
        self.model = load_model(modelName, custom_objects={'swish' : swish})
        self.grademodel=load_model(gradeModelName)
        # self.VSmodel = load_model(hivdModelName, custom_objects={'swish' : swish})
        #self.HIZmodel=load_model(hizModelName, custom_objects={'swish' : swish})


        # self.model._make_predict_function()
        # self.grademodel._make_predict_function()
        # self.graph = tf.get_default_graph()
        split=gradeModelName.split('_')
        if(len(split)>2):
            self.image_size = int(split[-2])
        else:
            self.image_size = 64
    def setGradeModel(self, gradeModelName):
        self.grademodel=load_model(gradeModelName)
        # self.grademodel._make_predict_function()
        # self.graph = tf.get_default_graph()
        split=gradeModelName.split('_')
        if(len(split)>2):
            self.image_size = int(split[-2])
        else:
            self.image_size = 64
    def modelPredict(self, fileName, savePath):
        print(fileName)
        self.fileName=fileName
        img = cv2.imread(fileName)
        imgTest = cv2.resize(img, (512, 512))
        try: 
            imgOri = imgTest.copy()
        except:
            print("check file")
            exit(0)

        imgTest = imgTest/255.
        imgTest = np.reshape(imgTest, (1, 512, 512, 3))


        #imgTest = load_img(fileName, grayscale=False, target_size=[512, 512])
        #imgTest = img_to_array(imgTest) 
        #imgTest = np.resize(img, (1, 512, 512, 3)).astype('float32')
        #imgTest /= 255
        
        # with self.graph.as_default():
        predicted = self.model.predict(imgTest, batch_size = 1, verbose=1)
        
        predicted = np.reshape(predicted, (512, 512, 3))
        predicted = np.argmax(predicted, axis = -1)
        predicted = np.reshape(predicted, (512, 512, 1)).astype(np.uint8)
        #cv2.imwrite('./test.jpg',predicted)
        label_seg = []
        first_need = np.zeros((imgTest.shape[1], imgTest.shape[2], 3), dtype=np.int)
        first_need[(predicted == np.array([1])).all(axis=2)] = [255, 255, 255]
        first_need = first_need.astype(np.uint8)
        
        second_need = np.zeros((imgTest.shape[1], imgTest.shape[2], 3), dtype=np.int)
        second_need[(predicted == np.array([2])).all(axis=2)] = [255, 255, 255]
        second_need = second_need.astype(np.uint8)
        #label_seg = np.concatenate(([first_need], [second_need]),axis = 0)
        label_seg = first_need
        imgBounding, mean, std, w, h ,grade,area,yolox,yoloy,yolow,yoloh= self.draw_and_get_bounding(label_seg, imgOri, savePath,imgOri.shape[0], imgOri.shape[1])
        #cv2.imshow("disc1",label_seg[0])
        #cv2.imshow("disc2",label_seg[1])
        return imgBounding, mean, std, w, h,grade,area,yolox,yoloy,yolow,yoloh

    def draw_and_get_bounding(self, imgSeg, imgOut, savePath,img_width, img_height):
        mean = []
        std = []
        W = []
        H = []
        grade =[] 
        Area=[]
        yolo_ctX =[] 
        yolo_ctY=[]
        yolo_width =[] 
        yolo_height=[]
        ROIsavePath = savePath
        if not os.path.exists(ROIsavePath): os.makedirs(ROIsavePath)
        count = 0
        imgCopy = imgOut.copy()
        #for i in range(imgSeg.shape[0]):
        imgGray = cv2.cvtColor(imgSeg, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.resize(imgGray, (512, 512), interpolation=cv2.INTER_CUBIC)
        imgBin = cv2.threshold(imgGray, 244, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(savePath,'./imgBin.jpg'),imgBin)
        imgMor = cv2.erode(imgBin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        imgMor = cv2.dilate(imgMor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        #_,contours, hier = cv2.findContours(imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    #oepncv3
        contours, hier = cv2.findContours(imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #opencv4
        # sacralMinX,sacralMinY = self.VSmodelPredict(self.fileName)
        # if(sacralMinY==[]):
        #    return #
        sacralMinY=512
        for c in contours:
            x, y, w, h = cv2.boundingRect(c) 
            cntX = x+0.5*w
            cntY = y+0.5*h

            #ellipse = cv2.fitEllipse(c)
            area = cv2.contourArea(c)
            if x>30 and y>30 and area > 300  and y<sacralMinY:
                    #print('TTTTTTT:'+str(x)+'_'+str(y)+'_'+str(w)+'_'+str(h))
                    #grade classification-----start
                try:
                    crop_img = imgCopy[y-18:y+h+18, x-30:x+w+30]  #原本椎節的中心點
                        #crop_img = imgCopy[int(y+0.5*h):(y+1.5*h), int(x-0.5*w):int(x+w*1.5)] #椎節往下位移找椎間盤
             
                    pres = cv2.resize(crop_img, (self.image_size, self.image_size))
                    pres = pres / 255.
                    pw, ph, pc = pres.shape
                    pinputImg = np.reshape(pres, (1, pw, ph, pc))
                    presult = np.argmax(self.grademodel.predict(pinputImg))+1

                    grade.append(presult)
                except():
                    grade.append(-1)
                        #grade classification-----end
                W.append(w)
                H.append(h)
                Area.append(area)
                yolo_ctX.append(cntX)
                yolo_ctY.append(cntY)
                yolo_width.append(int(x + w * 1.05)-int(x * 0.95))
                yolo_height.append(int(y + h * 1.05)-int(y * 0.95))

                mask = np.zeros((imgOut.shape[0], imgOut.shape[1], 1), dtype=np.uint8)
                cv2.fillPoly(mask, pts = [c], color=(255))
                imgGray = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
                m , st = cv2.meanStdDev(imgGray, mask = mask)
                me = cv2.mean(imgGray, mask = mask)
                mean.append(np.around(np.mean(m), decimals=2))
                std.append(np.around(np.mean(st), decimals=2))
                #text = 'disc ' + str(count)+' ('+str(presult)+')'
                text = 'disc ' + str(count)

                cv2.rectangle(imgOut, (int(x * 0.95), int(y * 0.95)), (int(x + w * 1.05), int(y + h * 1.05)), (0, 0, 255), 2)
                cv2.putText(imgOut, text, (x-165, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                ROI = imgCopy[max(0, int(y * 0.95)) : min(int(y + h * 1.05), imgOut.shape[0]), max(0, int(x * 0.95)) : min(int(x + w * 1.05), imgOut.shape[1])]
                sp = os.path.join(ROIsavePath, (self.curtime()+'_disc %d_predicted_%d.png' % (count,presult)))
                cv2.imwrite(sp, ROI)

                count += 1
        self.savePath=savePath
        img11=cv2.resize(imgCopy,(img_width,img_height))
        img22=cv2.resize(imgOut,(img_width,img_height))

        img_concate = np.concatenate((img11, img22),axis = 0)
        cv2.imwrite(os.path.join(savePath, ('_result.png' )), img_concate)
        
        cv2.imwrite(ROIsavePath+'_result.png', img_concate)
        cv2.imwrite(ROIsavePath+'_result_disc.png', img22)
        return imgOut, mean, std, W, H ,grade,Area,yolo_ctX,yolo_ctY,yolo_width,yolo_height

    def curtime(self):
        nowStamp = datetime.now()
        curtStr=str(nowStamp.year)+str(nowStamp.month)+str(nowStamp.day)+'_'+str(nowStamp.hour)+str(nowStamp.minute)+str(nowStamp.second)
        print(str(nowStamp.timestamp))
        tt= datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        return tt
    def VSmodelPredict(self,fileName):
        print(fileName)
        self.fileName=fileName
        img = cv2.imread(fileName)
        imgTest = cv2.resize(img, (512, 512))
        try: 
            imgOri = imgTest.copy()
        except:
            print("check file")
            exit(0)

        imgTest = imgTest/255.
        imgTest = np.reshape(imgTest, (1, 512, 512, 3))

        with self.graph.as_default():
            predicted = self.VSmodel.predict(imgTest, batch_size = 1, verbose=1)
            #predicted = self.model.predict(imgTest, batch_size = 1, verbose=1)
        
        predicted = np.reshape(predicted, (512, 512, 3))
        predicted = np.argmax(predicted, axis = -1)
        predicted = np.reshape(predicted, (512, 512, 1)).astype(np.uint8)

        second_need = np.zeros((imgTest.shape[1], imgTest.shape[2], 1), dtype=np.int)
        second_need[(predicted == np.array([2])).all(axis=2)] = 255
        imgBin = second_need.astype(np.uint8)
        cv2.imwrite('./Sacral.jpg',imgBin)

        #contours, hier = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #opencv4
        #_,contours, hier = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    #oepncv3
        contours, hier = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #opencv4
        cv2.imwrite('./Sacral_imgBin.jpg',imgBin)
        rtnX=[]
        rtnY=[]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c) 
            print(str(x)+','+str(y)+','+str(w)+','+str(h))
            rtnX.append(x)
            rtnY.append(y)
        print(rtnX)
        print(rtnY)
        return np.min(rtnX),np.min(rtnY)


# class EvaluteSignals(QtCore.QObject):
#     finished = QtCore.pyqtSignal()
#     error = QtCore.pyqtSignal(tuple)
#     result = QtCore.pyqtSignal(object)
#     progress = QtCore.pyqtSignal(object)

# class EvaluteThread(QtCore.QRunnable):
#     def __init__(self, fn, *args, **kwargs):
#         super(EvaluteThread, self).__init__()

#         # Store constructor arguments (re-used for processing)
#         self.fn = fn
#         self.args = args
#         self.kwargs = kwargs
#         self.signals = EvaluteSignals()

#         # Add the callback to our kwargs
#         self.kwargs['progress_callback'] = self.signals.progress

#     def run(self):
#         try:
#             result = self.fn(*self.args, **self.kwargs)
#         except:
#             traceback.print_exc()
#             exctype, value = sys.exc_info()[:2]
#             self.signals.error.emit((exctype, value, traceback.format_exc()))
#         else:
#             self.signals.result.emit(result)  # Return the result of the processing
#         finally:
#             self.signals.finished.emit()

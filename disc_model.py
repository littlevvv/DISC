###############
# env conda run --name FY114_CSCslag_3916 python your_script.py 
###############
from keras import backend as K  #2.6
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2, os

def swish(x):
    return (K.sigmoid(x) * x)

modelName = "/media/r300/1T/A30335/Disc/ITRI_CMS_R100_unet_residual_vgg16.hdf5"
model = load_model(modelName, custom_objects={'swish' : swish})
# model._make_predict_function()  # 在新版本的 Keras 中不再需要
fileName = "/media/r300/1T/A30335/Disc/1 (3).png"

img = cv2.imread(fileName)
imgTest = cv2.resize(img, (512, 512))
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

# 進行預測
predicted = model.predict(imgTest, batch_size = 1, verbose=1)

predicted = np.reshape(predicted, (512, 512, 3))
predicted = np.argmax(predicted, axis = -1)
predicted = np.reshape(predicted, (512, 512, 1)).astype(np.uint8)
#cv2.imwrite('./test.jpg',predicted)
label_seg = []
first_need = np.zeros((imgTest.shape[1], imgTest.shape[2], 3), dtype=np.int32)
first_need[(predicted == np.array([1])).all(axis=2)] = [255, 255, 255]
first_need = first_need.astype(np.uint8)

second_need = np.zeros((imgTest.shape[1], imgTest.shape[2], 3), dtype=np.int32)
second_need[(predicted == np.array([2])).all(axis=2)] = [255, 255, 255]
second_need = second_need.astype(np.uint8)
#label_seg = np.concatenate(([first_need], [second_need]),axis = 0)
label_seg = first_need

# 保存結果
savePath = "output_result.jpg"
cv2.imwrite(savePath, label_seg)
print(f"結果已保存到: {savePath}")

# 顯示結果 - 註解掉因為沒有顯示器
# cv2.imshow("Original", imgOri)
# cv2.imshow("Segmented", label_seg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# imgBounding, mean, std, w, h ,grade,area,yolox,yoloy,yolow,yoloh= self.draw_and_get_bounding(label_seg, imgOri, savePath,imgOri.shape[0], imgOri.shape[1])
#cv2.imshow("disc1",label_seg[0])
#cv2.imshow("disc2",label_seg[1])
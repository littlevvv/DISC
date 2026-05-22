import gradio as gr
import cv2
import numpy as np
import os
from keras import backend as K
from keras.models import load_model

def swish(x):
    return (K.sigmoid(x) * x)

# 載入模型
print("正在載入模型...")
modelName = "/media/r300/1T/A30335/Disc/ITRI_CMS_R100_unet_residual_vgg16.hdf5"
model = load_model(modelName, custom_objects={'swish': swish})
print("模型載入完成！")

def process_image(input_image):
    """
    處理上傳的圖片並返回分割結果
    """
    try:
        if input_image is None:
            return None, None, None, "請先上傳圖片"
        
        # 如果輸入是文件路徑
        if isinstance(input_image, str):
            img = cv2.imread(input_image)
        else:
            # 將 PIL 圖片轉為 OpenCV 格式
            img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
        if img is None:
            return None, None, None, "錯誤：無法讀取圖片"
        
        # 保存原圖
        imgOri = img.copy()
        
        # 預處理圖片
        imgTest = cv2.resize(img, (512, 512))
        imgTest = imgTest / 255.0
        imgTest = np.reshape(imgTest, (1, 512, 512, 3))
        
        # 進行預測
        predicted = model.predict(imgTest, batch_size=1, verbose=0)
        
        # 後處理
        predicted = np.reshape(predicted, (512, 512, 3))
        predicted = np.argmax(predicted, axis=-1)
        predicted = np.reshape(predicted, (512, 512, 1)).astype(np.uint8)
        
        # 創建分割結果圖像
        first_need = np.zeros((512, 512, 3), dtype=np.int32)
        first_need[(predicted == np.array([1])).all(axis=2)] = [255, 255, 255]
        first_need = first_need.astype(np.uint8)
        
        second_need = np.zeros((512, 512, 3), dtype=np.int32)
        second_need[(predicted == np.array([2])).all(axis=2)] = [255, 255, 255]
        second_need = second_need.astype(np.uint8)
        
        # 將原圖調整到512x512
        imgOri_resized = cv2.resize(imgOri, (512, 512))
        
        # 轉換為 RGB 格式給 Gradio 顯示
        imgOri_rgb = cv2.cvtColor(imgOri_resized, cv2.COLOR_BGR2RGB)
        first_need_rgb = cv2.cvtColor(first_need, cv2.COLOR_BGR2RGB)
        second_need_rgb = cv2.cvtColor(second_need, cv2.COLOR_BGR2RGB)
        
        info_text = "分割完成！\n白色區域代表檢測到的椎間盤組織"
        
        return imgOri_rgb, first_need_rgb, second_need_rgb, info_text
        
    except Exception as e:
        return None, None, None, f"處理錯誤：{str(e)}"

# 創建 Gradio 介面
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="上傳醫學影像"),
    outputs=[
        gr.Image(type="numpy", label="原始圖像"),
        gr.Image(type="numpy", label="第一類分割結果"),
        gr.Image(type="numpy", label="第二類分割結果"),
        gr.Textbox(label="處理狀態")
    ],
    title="椎間盤分割系統",
    description="上傳一張醫學影像，系統將自動進行椎間盤分割分析",
    examples=[
        ["/media/r300/1T/A30335/Disc/1 (3).png"]
    ] if os.path.exists("/media/r300/1T/A30335/Disc/1 (3).png") else None
)

if __name__ == "__main__":
    print("啟動 Gradio 介面...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False
    )

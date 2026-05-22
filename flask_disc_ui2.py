from flask import Flask, request, render_template, jsonify, send_file, url_for
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
from keras import backend as K
from keras.models import load_model
from datetime import datetime

def swish(x):
    return (K.sigmoid(x) * x)

# 建立 Flask 應用
app = Flask(__name__)

# 載入所有模型
print("正在載入模型...")
modelName = "/media/r300/1T/A30335/Disc/ITRI_CMS_R100_unet_residual_vgg16.hdf5"
gradeModelName = "/media/r300/1T/A30335/Disc/past/grademodel/ITRI_CMSR100_DenseNet121_50_128_loss.hdf5"

model = load_model(modelName, custom_objects={'swish': swish})
try:
    grademodel = load_model(gradeModelName)
    grade_model_available = True
    # 從模型名稱解析圖像尺寸
    split = gradeModelName.split('_')
    if len(split) > 2:
        image_size = int(split[-2])
    else:
        image_size = 64
    print(f"等級分類模型載入成功，圖像尺寸: {image_size}")
except:
    grademodel = None
    grade_model_available = False
    image_size = 64
    print("等級分類模型載入失敗，將只進行分割")

print("模型載入完成！")

def process_image(img_array):
    """
    處理圖片並返回分割結果，參考 predict.py 的完整實作
    """
    try:
        # 預處理圖片
        imgOri = img_array.copy()
        imgTest = cv2.resize(img_array, (512, 512))
        imgCopy = imgTest.copy()
        imgTest = imgTest / 255.0
        imgTest = np.reshape(imgTest, (1, 512, 512, 3))
        
        # 進行語意分割預測
        predicted = model.predict(imgTest, batch_size=1, verbose=0)
        
        # 後處理分割結果
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
        
        # 使用第一類分割結果進行邊界檢測和等級分類
        label_seg = first_need
        imgGray = cv2.cvtColor(label_seg, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.resize(imgGray, (512, 512), interpolation=cv2.INTER_CUBIC)
        imgBin = cv2.threshold(imgGray, 244, 255, cv2.THRESH_BINARY)[1]
        
        # 形態學處理
        imgMor = cv2.erode(imgBin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        imgMor = cv2.dilate(imgMor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        
        # 尋找輪廓
        contours, hier = cv2.findContours(imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 創建結果圖像
        imgOut = imgCopy.copy()
        count = 0
        
        # 處理每個檢測到的椎間盤區域
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # 過濾條件：位置和面積
            if x > 30 and y > 30 and area > 300 and y < 512:
                # 等級分類
                grade_result = -1
                if grade_model_available and grademodel is not None:
                    try:
                        # 提取ROI進行等級分類
                        crop_img = imgCopy[y-18:y+h+18, x-30:x+w+30]
                        if crop_img.size > 0:
                            pres = cv2.resize(crop_img, (image_size, image_size))
                            pres = pres / 255.0
                            pw, ph, pc = pres.shape
                            pinputImg = np.reshape(pres, (1, pw, ph, pc))
                            grade_result = np.argmax(grademodel.predict(pinputImg, verbose=0)) + 1
                    except Exception as e:
                        print(f"等級分類錯誤: {e}")
                        grade_result = -1
                
                # 在結果圖像上繪製邊界框和標籤
                text = f'disc {count}'
                if grade_result > 0:
                    text += f' (Grade {grade_result})'
                
                cv2.rectangle(imgOut, (int(x * 0.95), int(y * 0.95)), 
                             (int(x + w * 1.05), int(y + h * 1.05)), (0, 0, 255), 2)
                cv2.putText(imgOut, text, (x-165, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                count += 1
        
        # 創建最終的等級分類結果圖像
        # 將原圖和標註結果合併
        img_concate = np.concatenate((imgCopy, imgOut), axis=0)
        
        # 保存等級分類結果為 _result.jpg
        cv2.imwrite('_result.jpg', img_concate)
        cv2.imwrite('_result_disc.jpg', imgOut)
        
        # 將原圖調整到512x512
        imgOri_resized = cv2.resize(imgOri, (512, 512))
        
        return imgOri_resized, first_need, imgOut
        
    except Exception as e:
        print(f"處理錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def img_to_base64(img_array):
    """
    將 numpy 陣列轉換為 base64 字符串
    """
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    """
    主頁面
    <a href="/paper" download>📥 下載論文PDF</a>
    <a href="/paper_twi" download>📥 下載TWI專利</a>
    <a href="/paper_us" download>📥 下載US專利</a>
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>椎間盤分割系統</title>
        <meta charset="UTF-8">
        <link rel="icon" href="/favicon.ico" type="image/x-icon">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin-bottom: 30px; border-radius: 5px; }
            .upload-area:hover { border-color: #007bff; background-color: #f8f9fa; }
            input[type="file"] { margin: 20px 0; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            .results { margin-top: 30px; }
            .result-row { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
            .result-item { flex: 1; min-width: 250px; }
            .result-item h3 { color: #333; margin-bottom: 10px; }
            .result-item img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
            .status { padding: 15px; margin: 20px 0; border-radius: 5px; }
            .status.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .loading { display: none; text-align: center; margin: 20px 0; }
            .paper-links { text-align: center; margin: 20px 0; }
            .paper-links a { color: #007bff; text-decoration: none; margin: 0 15px; padding: 8px 16px; border: 1px solid #007bff; border-radius: 4px; transition: all 0.3s; }
            .paper-links a:hover { background-color: #007bff; color: white; }
            .paper-info { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px; color: #495057; }
            .title-with-icon { display: flex; align-items: center; justify-content: center; gap: 15px; }
            .title-with-icon img { width: 48px; height: 48px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title-with-icon">
                <img src="/favicon.ico" alt="ITRI Logo">
                椎間盤辨識系統
            </h1>
            
            
            
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                上傳一張醫學影像，系統將自動進行椎間盤分割分析
            </p>
            
            <div class="upload-area">
                <p>點擊選擇文件或拖拽圖片到此處</p>
                <input type="file" id="imageInput" accept="image/*" />
                <br>
                <button onclick="processImage()">開始分析</button>
            </div>
            
            <div class="loading" id="loading">
                <p>正在處理中，請稍候...</p>
            </div>
            
            <div id="status"></div>
            
            <div class="results" id="results" style="display: none;">
                <h2>分析結果</h2>
                <div class="result-row">
                    <div class="result-item">
                        <h3>原始圖像</h3>
                        <img id="originalImg" />
                    </div>
                    <div class="result-item">
                        <h3>UNET 語意切割結果</h3>
                        <img id="firstImg" />
                    </div>
                    <div class="result-item">
                        <h3>椎間盤等級分類結果</h3>
                        <img id="secondImg" />
                    </div>
                </div>
                <p style="color: #666; margin-top: 20px;">
                    <strong>說明：</strong><br>
                    • UNET 語意切割：白色區域代表檢測到的椎間盤組織<br>
                    • 椎間盤等級分類：紅色邊界框標示椎間盤位置，並顯示等級分類結果<br>
                    • 完整分析結果已保存為 _result.jpg 文件（包含原圖和分析結果的合併圖像）<br>
                    • 椎間盤標註結果已保存為 _result_disc.jpg 文件
                </p>
                <div class="paper-info">
                <strong>📄 相關研究論文：</strong>脊椎MRI影像之椎間盤切割與分類技術, “Multitask Deep Learning for Segmentation and Lumbosacral Spine Inspection,” IEEE Transactions on Instrumentation and Measurement, Vol. 71, Jul. 2022. <br>
                <strong>📋 相關專利文獻：</strong>TW202324441A, US12136485
            </div>
            
            <div class="paper-links">
                <a href="/view_paper" target="_blank">📖 線上閱讀中文論文</a>
                <a href="/view_paper_ieee" target="_blank">📖 線上閱讀IEEE論文</a>
                
                <a href="/view_paper_twi" target="_blank">📖 TWI專利文獻</a>
                
                <a href="/view_paper_us" target="_blank">📖 US專利文獻</a>
                
            </div>
            </div>
        </div>

        <script>
            function processImage() {
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    showStatus('請先選擇一張圖片', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', file);
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                showStatus('', '');
                
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    
                    if (data.success) {
                        document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original;
                        document.getElementById('firstImg').src = 'data:image/png;base64,' + data.first;
                        document.getElementById('secondImg').src = 'data:image/png;base64,' + data.second;
                        document.getElementById('results').style.display = 'block';
                        showStatus('分析完成！', 'success');
                    } else {
                        showStatus('處理錯誤：' + data.error, 'error');
                    }
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    showStatus('網路錯誤：' + error.message, 'error');
                });
            }
            
            function showStatus(message, type) {
                const statusDiv = document.getElementById('status');
                if (message) {
                    statusDiv.innerHTML = message;
                    statusDiv.className = 'status ' + type;
                    statusDiv.style.display = 'block';
                } else {
                    statusDiv.style.display = 'none';
                }
            }
            
            // 文件拖拽功能
            const uploadArea = document.querySelector('.upload-area');
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#007bff';
                uploadArea.style.backgroundColor = '#f8f9fa';
            });
            
            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.backgroundColor = 'transparent';
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = '#ccc';
                uploadArea.style.backgroundColor = 'transparent';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('imageInput').files = files;
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process():
    """
    處理上傳的圖片
    """
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '沒有上傳文件'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '沒有選擇文件'})
        
        # 讀取圖片
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': '無法讀取圖片文件'})
        
        # 處理圖片
        original, first, second = process_image(img)
        
        if original is None:
            return jsonify({'success': False, 'error': '圖片處理失敗'})
        
        # 轉換為 base64
        original_b64 = img_to_base64(original)
        first_b64 = img_to_base64(first)
        second_b64 = img_to_base64(second)
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'first': first_b64,
            'second': second_b64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/paper')
def download_paper():
    """
    提供研究論文下載服務
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/計量文章_脊椎MRI影像之椎間盤切割與分類技術_劉曉薇 賴程威_0610 1.pdf"
        return send_file(paper_path, 
                        as_attachment=True, 
                        download_name="椎間盤切割與分類技術論文.pdf",
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法提供論文下載 - {str(e)}", 404

@app.route('/view_paper')
def view_paper():
    """
    在瀏覽器中查看研究論文
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/計量文章_脊椎MRI影像之椎間盤切割與分類技術_劉曉薇 賴程威_0610 1.pdf"
        return send_file(paper_path, 
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法顯示論文 - {str(e)}", 404

@app.route('/paper_twi')
def download_paper_twi():
    """
    提供TWI專利文獻下載服務
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/TW202324441A.pdf"
        return send_file(paper_path, 
                        as_attachment=True, 
                        download_name="TW202324441A.pdf",
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法提供TWI專利下載 - {str(e)}", 404

@app.route('/view_paper_twi')
def view_paper_twi():
    """
    在瀏覽器中查看TWI專利文獻
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/TWI722264B.pdf"
        return send_file(paper_path, 
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法顯示TWI專利 - {str(e)}", 404

@app.route('/paper_us')
def download_paper_us():
    """
    提供US專利文獻下載服務
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/US12136485.pdf"
        return send_file(paper_path, 
                        as_attachment=True, 
                        download_name="US12136485.pdf",
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法提供US專利下載 - {str(e)}", 404

@app.route('/view_paper_us')
def view_paper_us():
    """
    在瀏覽器中查看US專利文獻
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/US12136485.pdf"
        return send_file(paper_path, 
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法顯示US專利 - {str(e)}", 404
    
@app.route('/view_paper_ieee')
def view_paper_ieee():
    """
    在瀏覽器中查看IEEE專利文獻
    """
    try:
        paper_path = "/media/r300/1T/A30335/Disc/paperReference/Multitask_Deep_Learning_for_Segmentation_and_Lumbosacral_Spine_Inspection.pdf"
        return send_file(paper_path, 
                        mimetype='application/pdf')
    except Exception as e:
        return f"錯誤：無法顯示IEEE專利 - {str(e)}", 404
@app.route('/favicon.ico')
def favicon():
    """
    提供網站圖標
    """
    try:
        return send_file("/media/r300/1T/A30335/Disc/ico/ITRI128.ico", 
                        mimetype='image/vnd.microsoft.icon')
    except Exception as e:
        return f"錯誤：無法載入圖標 - {str(e)}", 404

if __name__ == '__main__':
    print("啟動 Flask 網頁應用...")
    print("模型已載入，準備就緒！")
    print("請在瀏覽器中訪問: http://localhost:6060")
    app.run(host='0.0.0.0', port=6060, debug=False)

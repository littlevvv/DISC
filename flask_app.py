###############
#
#######################
# cd /media/r300/1T/A30335/Disc && conda activate disc && python flask_app.py
#######################

from flask import Flask, request, render_template, jsonify, send_file, url_for, Response
import cv2
import numpy as np
import os
import base64
import io
import json
import subprocess
import sys
from queue import Empty
from PIL import Image
from keras import backend as K
from keras.models import load_model
from datetime import datetime

def swish(x):
    return (K.sigmoid(x) * x)

# 建立 Flask 應用
app = Flask(__name__)

# 載入深度學習模型
print("正在載入椎間盤分割模型...")
modelName = "/media/r300/1T/A30335/Disc/ITRI_CMS_R100_unet_residual_vgg16.hdf5"
model = load_model(modelName, custom_objects={'swish': swish})
print("椎間盤分割模型載入完成！")

# 初始化 chatbot（延遲載入）
chatbot_instances = {}

def initialize_chatbot():
    """
    初始化 chatbot - 使用 chatbot conda 環境
    """
    try:
        # 使用 subprocess 在 chatbot 環境中執行 python 代碼
        print("正在初始化醫療小幫手...")
        return True
    except Exception as e:
        print(f"Chatbot 初始化失敗: {str(e)}")
        return False

def get_chatbot_response_via_subprocess(message):
    """
    通過 subprocess 在 chatbot 環境中獲取回應
    """
    try:
        # 創建一個臨時的 Python 腳本來在 chatbot 環境中執行
        script_content = f'''
import sys
sys.path.append("/media/r300/1T/A30335/Disc")
from chatbot import ChatBot
import json

try:
    chatbot = ChatBot(
        model_type="smart-factory",
        gpt_model="phi4",
        log_path="/media/r300/1T/A30335/Disc/chatlog",
        isref=0
    )
    
    queue, callback = chatbot.get_streaming_response("{message}")
    
    complete_response = ""
    while True:
        try:
            token = queue.get(timeout=30)
            if token is None:
                complete_response = callback.get_complete_response()
                break
            complete_response += token
        except:
            break
    
    result = {{"success": True, "response": complete_response}}
    print(json.dumps(result))
    
except Exception as e:
    result = {{"success": False, "error": str(e)}}
    print(json.dumps(result))
'''
        
        # 寫入臨時文件
        temp_script = "/tmp/chatbot_temp.py"
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 在 chatbot 環境中執行
        result = subprocess.run([
            'conda', 'run', '-n', 'chatbot', 'python', temp_script
        ], capture_output=True, text=True, timeout=60)
        
        # 清理臨時文件
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        if result.returncode == 0:
            response_data = json.loads(result.stdout.strip())
            return response_data
        else:
            return {"success": False, "error": f"Subprocess error: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "回應超時"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def process_image(img_array):
    """
    處理圖片並返回分割結果 - 基於 predict.py 的實作
    """
    try:
        # 預處理圖片
        imgOri = img_array.copy()
        imgTest = cv2.resize(img_array, (512, 512))
        imgTest = imgTest / 255.0
        imgTest = np.reshape(imgTest, (1, 512, 512, 3))
        
        # 進行預測
        predicted = model.predict(imgTest, batch_size=1, verbose=0)
        
        # 後處理
        predicted = np.reshape(predicted, (512, 512, 3))
        predicted = np.argmax(predicted, axis=-1)
        predicted = np.reshape(predicted, (512, 512, 1)).astype(np.uint8)
        
        # 創建第一類分割結果
        first_need = np.zeros((512, 512, 3), dtype=np.int32)
        first_need[(predicted == np.array([1])).all(axis=2)] = [255, 255, 255]
        first_need = first_need.astype(np.uint8)
        
        # 創建第二類分割結果
        second_need = np.zeros((512, 512, 3), dtype=np.int32)
        second_need[(predicted == np.array([2])).all(axis=2)] = [255, 255, 255]
        second_need = second_need.astype(np.uint8)
        
        # 創建椎間盤檢測結果（結合邊界檢測）
        imgSeg = first_need.copy()
        imgGray = cv2.cvtColor(imgSeg, cv2.COLOR_BGR2GRAY)
        imgBin = cv2.threshold(imgGray, 244, 255, cv2.THRESH_BINARY)[1]
        
        # 形態學處理
        imgMor = cv2.erode(imgBin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        imgMor = cv2.dilate(imgMor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        
        # 尋找輪廓 (OpenCV 4.x 版本)
        contours, hier = cv2.findContours(imgMor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 創建輸出圖像
        imgOut = cv2.resize(imgOri, (512, 512))
        
        # 在圖像上繪製檢測結果
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            # 過濾小區域
            if area > 500:  # 最小面積閾值
                # 繪製邊界框
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # 添加標籤
                label = f"Disc: {int(area)}"
                cv2.putText(imgOut, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 保存結果圖像
        cv2.imwrite('_result.jpg', first_need)
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
    主頁面 - 使用模板渲染
    """
    return render_template('index.html', current_year=datetime.now().year)

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

@app.route('/chat', methods=['POST'])
def chat():
    """
    醫療小幫手聊天接口
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': '請提供問題內容'})
        
        message = data['message'].strip()
        if not message:
            return jsonify({'success': False, 'error': '問題不能為空'})
        
        print(f"收到問題: {message}")
        
        # 通過 subprocess 獲取 chatbot 回應
        response_data = get_chatbot_response_via_subprocess(message)
        
        if response_data['success']:
            return jsonify({
                'success': True,
                'response': response_data['response']
            })
        else:
            return jsonify({
                'success': False,
                'error': response_data.get('error', '未知錯誤')
            })
            
    except Exception as e:
        print(f"Chat 錯誤: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """
    醫療小幫手串流聊天接口
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': '請提供問題內容'})
        
        message = data['message'].strip()
        if not message:
            return jsonify({'success': False, 'error': '問題不能為空'})
        
        def generate_stream():
            try:
                # 創建串流回應的腳本
                script_content = f'''
import sys
sys.path.append("/media/r300/1T/A30335/Disc")
from chatbot import ChatBot
import json
from queue import Empty

try:
    chatbot = ChatBot(
        model_type="smart-factory",
        gpt_model="phi4",
        log_path="/media/r300/1T/A30335/Disc/chatlog",
        isref=0
    )
    
    queue, callback = chatbot.get_streaming_response("{message}")
    
    while True:
        try:
            token = queue.get(timeout=30)
            if token is None:
                complete_response = callback.get_complete_response()
                final_data = {{
                    "status": "complete",
                    "response": complete_response,
                    "is_final": True
                }}
                print("STREAM:" + json.dumps(final_data))
                break
                
            data = {{
                "status": "streaming",
                "token": token,
                "is_final": False
            }}
            print("STREAM:" + json.dumps(data))
        except Empty:
            error_data = {{
                "status": "error", 
                "message": "回應超時",
                "is_final": True
            }}
            print("STREAM:" + json.dumps(error_data))
            break
            
except Exception as e:
    error_data = {{"status": "error", "message": str(e), "is_final": True}}
    print("STREAM:" + json.dumps(error_data))
'''
                
                temp_script = "/tmp/chatbot_stream.py"
                with open(temp_script, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                
                # 執行串流腳本
                process = subprocess.Popen([
                    'conda', 'run', '-n', 'chatbot', 'python', temp_script
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True, bufsize=1, universal_newlines=True)
                
                for line in process.stdout:
                    if line.startswith("STREAM:"):
                        stream_data = line[7:].strip()  # 移除 "STREAM:" 前綴
                        yield f"data: {stream_data}\n\n"
                
                # 清理
                process.wait()
                if os.path.exists(temp_script):
                    os.remove(temp_script)
                    
            except Exception as e:
                error_data = {
                    "status": "error",
                    "message": str(e),
                    "is_final": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(generate_stream(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("啟動 Flask 網頁應用...")
    print("模型已載入，準備就緒！")
    print("請在瀏覽器中訪問: http://localhost:5001")
    app.run(host='0.0.0.0', port=5002, debug=False)

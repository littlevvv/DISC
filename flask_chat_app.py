#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask 聊天機器人應用程式
基於 chatbot_test.py 創建的獨立聊天 UI
運行命令: cd /media/r300/1T/A30335/Disc && conda activate chatbot && python flask_chat_app.py
"""

from flask import Flask, request, render_template, jsonify, Resp    
    print(f"🚀 應用程式啟動在: http://localhost:5003")
    print("💡 確保您正在使用 'chatbot' conda 環境")
    print("=" * 60)
    
    # 啟動 Flask 應用
    app.run(host='0.0.0.0', port=5003, debug=True)mport json
import os
import sys
from queue import Empty
from datetime import datetime
from chatbot import ChatBot
import zhconv

# 建立 Flask 應用
app = Flask(__name__)

# 全域 chatbot 實例
chatbot_instance = None

def convert_text(text, conversion_type='s2tw'):
    """
    使用 zhconv 轉換文字
    
    Args:
        text (str): 要轉換的文字
        conversion_type (str): 轉換類型
            - 's2t': 簡體轉繁體
            - 't2s': 繁體轉簡體  
            - 's2tw': 簡體轉台灣繁體
            - 't2tw': 繁體轉台灣繁體
            - 's2hk': 簡體轉香港繁體
            - 't2hk': 繁體轉香港繁體
    
    Returns:
        str: 轉換後的文字
    """
    try:
        if conversion_type == 's2t' or conversion_type == 's2tw':
            # 簡體轉繁體
            return zhconv.convert(text, 'zh-tw')
        elif conversion_type == 't2s' or conversion_type == 'tw2s':
            # 繁體轉簡體
            return zhconv.convert(text, 'zh-cn')
        elif conversion_type == 's2hk':
            # 簡體轉香港繁體
            return zhconv.convert(text, 'zh-hk')
        elif conversion_type == 't2tw':
            # 繁體轉台灣繁體
            return zhconv.convert(text, 'zh-tw')
        elif conversion_type == 't2hk':
            # 繁體轉香港繁體
            return zhconv.convert(text, 'zh-hk')
        else:
            # 不轉換，返回原文
            return text
    except Exception as e:
        print(f"zhconv 轉換錯誤: {str(e)}")
        return text

def initialize_chatbot():
    """
    初始化聊天機器人
    """
    global chatbot_instance
    try:
        print("正在初始化醫療聊天機器人...")
        chatbot_instance = ChatBot(
            model_type="smart-factory",
            gpt_model="llama3:latest",
            log_path="/media/r300/1T/A30335/Disc/chatlog",
            isref=0
        )
        print("聊天機器人初始化完成！")
        return True
    except Exception as e:
        print(f"聊天機器人初始化失敗: {str(e)}")
        return False

@app.route('/')
def index():
    """
    主頁面 - 聊天界面
    """
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    處理聊天請求 - 非串流版本，支援文字轉換
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        convert_type = data.get('convert_type', 's2tw')  # 預設簡體轉台灣繁體
        
        if not message:
            return jsonify({'error': '請輸入有效的問題'}), 400
        
        if not chatbot_instance:
            return jsonify({'error': '聊天機器人未初始化'}), 500
        
        print(f"收到問題: {message}")
        print(f"轉換類型: {convert_type}")
        
        # 獲取串流回應
        queue, callback = chatbot_instance.get_streaming_response(message)
        
        # 收集完整回應
        complete_response = ""
        while True:
            try:
                token = queue.get(timeout=30)  # 30秒超時
                if token is None:  # End of stream
                    complete_response = callback.get_complete_response()
                    break
                complete_response += token
            except Empty:
                complete_response = "抱歉，回應超時。請重新嘗試您的問題。"
                break
        
        # 轉換回應文字
        converted_response = convert_text(complete_response, convert_type)
        
        return jsonify({
            'response': converted_response,
            'original_response': complete_response,
            'conversion_type': convert_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        print(f"聊天處理錯誤: {str(e)}")
        return jsonify({'error': '處理您的請求時發生錯誤'}), 500

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """
    處理聊天請求 - 串流版本 (Server-Sent Events)，支援文字轉換
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        convert_type = data.get('convert_type', 's2tw')  # 預設簡體轉台灣繁體
        
        if not message:
            def error_generator():
                yield f"data: {json.dumps({'error': '請輸入有效的問題'}, ensure_ascii=False)}\n\n"
            return Response(error_generator(), mimetype='text/plain')
        
        if not chatbot_instance:
            def error_generator():
                yield f"data: {json.dumps({'error': '聊天機器人未初始化'}, ensure_ascii=False)}\n\n"
            return Response(error_generator(), mimetype='text/plain')
        
        print(f"收到串流問題: {message}")
        print(f"轉換類型: {convert_type}")
        
        def generate_response():
            try:
                queue, callback = chatbot_instance.get_streaming_response(message)
                
                while True:
                    try:
                        token = queue.get(timeout=30)  # 30秒超時
                        if token is None:  # End of stream
                            complete_response = callback.get_complete_response()
                            
                            # 轉換完整回應
                            converted_response = convert_text(complete_response, convert_type)
                            
                            final_data = {
                                'status': 'complete',
                                'response': converted_response,
                                'original_response': complete_response,
                                'conversion_type': convert_type,
                                'is_final': True,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                            break
                            
                        # 對每個 token 也進行轉換（即時轉換）
                        converted_token = convert_text(token, convert_type)
                        data = {
                            'status': 'streaming',
                            'token': converted_token,
                            'original_token': token,
                            'is_final': False
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    except Empty:
                        # 如果隊列超時，發送錯誤消息
                        error_data = {
                            'status': 'error',
                            'message': '回應超時，請重新嘗試',
                            'is_final': True
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        break
            except Exception as e:
                error_data = {
                    'status': 'error',
                    'message': f'處理請求時發生錯誤: {str(e)}',
                    'is_final': True
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return Response(generate_response(), mimetype='text/event-stream')
        
    except Exception as e:
        print(f"串流聊天處理錯誤: {str(e)}")
        def error_generator():
            yield f"data: {json.dumps({'error': f'處理您的請求時發生錯誤: {str(e)}'}, ensure_ascii=False)}\n\n"
        return Response(error_generator(), mimetype='text/plain')

@app.route('/health')
def health():
    """
    健康檢查端點
    """
    return jsonify({
        'status': 'healthy',
        'chatbot_initialized': chatbot_instance is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    """
    重置聊天對話
    """
    try:
        global chatbot_instance
        if chatbot_instance:
            # 重新初始化聊天機器人以清除對話歷史
            initialize_chatbot()
        return jsonify({'message': '聊天對話已重置'})
    except Exception as e:
        return jsonify({'error': f'重置失敗: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Flask 醫療聊天機器人應用程式")
    print("=" * 60)
    
    # 初始化聊天機器人
    if initialize_chatbot():
        print("✅ 聊天機器人初始化成功")
    else:
        print("❌ 聊天機器人初始化失敗")
        sys.exit(1)
    
    print(f"🚀 應用程式啟動在: http://localhost:5005")
    print("💡 確保您正在使用 'chatbot' conda 環境")
    print("=" * 60)
    
    # 啟動 Flask 應用
    app.run(host='0.0.0.0', port=5005, debug=True)

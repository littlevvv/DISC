from chatbot import ChatBot
import os
import json
from queue import Empty

def test_chatbot_streaming():
    """
    測試聊天機器人的串流回應功能
    """
    chatbot_instances = {}
    chatbot_instances['disc'] = ChatBot(
        model_type="smart-factory",
        gpt_model="phi4",
        log_path="/media/r300/1T/A30335/Disc/chatlog",
        isref=0
    )
    
    chatbot = chatbot_instances['disc']
    message = 'what is intervertebral disc degeneration?'
    
    print(f"詢問問題: {message}")
    print("等待回應...")
    print("-" * 50)
    
    queue, callback = chatbot.get_streaming_response(message)
    
    complete_response = ""
    
    while True:
        try:
            token = queue.get(timeout=30)  # 30秒超時
            if token is None:  # End of stream
                complete_response = callback.get_complete_response()
                print("\n" + "=" * 50)
                print("完整回應:")
                print(complete_response)
                break
                
            # 即時顯示每個token
            print(token, end='', flush=True)
            complete_response += token
            
        except Empty:
            print("\n錯誤：回應超時")
            break

def get_streaming_generator(message):
    """
    生成器函數，用於串流回應
    """
    chatbot_instances = {}
    chatbot_instances['disc'] = ChatBot(
        model_type="smart-factory",
        gpt_model="llama3:latest",
        log_path="/media/r300/1T/A30335/Disc/chatlog",
        isref=0
    )
    
    chatbot = chatbot_instances['disc']
    queue, callback = chatbot.get_streaming_response(message)
    
    while True:
        try:
            token = queue.get(timeout=30)  # 30秒超時
            if token is None:  # End of stream
                complete_response = callback.get_complete_response()
                final_data = {
                    'status': 'complete',
                    'response': complete_response,
                    'is_final': True
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                break
                
            data = {
                'status': 'streaming',
                'token': token,
                'is_final': False
            }
            yield f"data: {json.dumps(data)}\n\n"
        except Empty:
            # 如果隊列超時，發送錯誤消息
            error_data = {
                'status': 'error',
                'message': '回應超時',
                'is_final': True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            break

if __name__ == "__main__":
    # 測試聊天機器人功能
    test_chatbot_streaming()
                        
 
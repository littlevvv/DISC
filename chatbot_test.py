####
#conda activate chatbot && pip install zhconv
###
from chatbot import ChatBot
import os
import json
from queue import Empty
import zhconv

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
            print(f"警告：不支援的轉換類型 '{conversion_type}'，返回原文")
            return text
    except Exception as e:
        print(f"zhconv 轉換錯誤: {str(e)}")
        return text

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
    message = '如何避免久坐問題? 以及亞健康影響'
    
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
                
                # 使用 zhconv 轉換回應
                print("\n" + "=" * 50)
                print("zhconv 轉換結果:")
                converted_response = convert_text(complete_response, 's2tw')
                print(f"簡體轉台灣繁體: {converted_response}")
                
                # 也可以嘗試其他轉換
                print(f"簡體轉繁體: {convert_text(complete_response, 's2t')}")
                print(f"簡體轉香港繁體: {convert_text(complete_response, 's2hk')}")
                
                # 如果原文是繁體，也可以轉簡體測試
                print(f"繁體轉簡體: {convert_text(complete_response, 't2s')}")
                break
                
            # 即時顯示每個token
            print(token, end='', flush=True)
            complete_response += token
            
        except Empty:
            print("\n錯誤：回應超時")
            break

def get_streaming_generator(message, conversion_type='s2tw'):
    """
    生成器函數，用於串流回應，支援 OpenCC 轉換
    
    Args:
        message (str): 輸入訊息
        conversion_type (str): OpenCC 轉換類型，預設為 's2tw'
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
                
                # 使用 zhconv 轉換完整回應
                converted_response = convert_text(complete_response, conversion_type)
                
                final_data = {
                    'status': 'complete',
                    'response': converted_response,
                    'original_response': complete_response,
                    'conversion_type': conversion_type,
                    'is_final': True
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                break
                
            # 對每個 token 也進行轉換（可選）
            converted_token = convert_text(token, conversion_type)
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
                'message': '回應超時',
                'is_final': True
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            break

def test_opencc_conversion():
    """
    測試 zhconv 轉換功能
    """
    print("=" * 60)
    print("zhconv 中文轉換測試")
    print("=" * 60)
    
    # 測試文字
    test_texts = [
        "椎间盘退变是一种常见的脊柱疾病。",
        "核磁共振成像可以清楚地显示椎间盘的结构。",
        "腰椎间盘突出症会导致下肢疼痛和麻木。",
        "保守治疗包括物理治疗、药物治疗和康复训练。"
    ]
    
    conversion_types = ['s2t', 's2tw', 's2hk', 't2s', 't2tw', 't2hk']
    conversion_names = {
        's2t': '簡體轉繁體',
        's2tw': '簡體轉台灣繁體', 
        's2hk': '簡體轉香港繁體',
        't2s': '繁體轉簡體',
        't2tw': '繁體轉台灣繁體',
        't2hk': '繁體轉香港繁體'
    }
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n測試文字 {i}: {text}")
        print("-" * 40)
        
        for conv_type in conversion_types:
            converted = convert_text(text, conv_type)
            print(f"{conversion_names[conv_type]}: {converted}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("選擇測試模式:")
    print("1. 聊天機器人功能測試")
    print("2. OpenCC 轉換測試")
    print("3. 同時測試兩個功能")
    
    choice = input("請輸入選擇 (1/2/3): ").strip()
    
    if choice == '1':
        test_chatbot_streaming()
    elif choice == '2':
        test_opencc_conversion()
    elif choice == '3':
        test_opencc_conversion()
        print("\n" + "=" * 60)
        print("開始聊天機器人測試...")
        print("=" * 60)
        test_chatbot_streaming()
    else:
        print("無效選擇，執行預設的聊天機器人測試")
        test_chatbot_streaming()
                        
 
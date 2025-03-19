from pymilvus import MilvusClient
import pymilvus
import sentence_transformers
import speech_recognition as sr
import pyttsx3
import json
import numpy as np
from transformers import pipeline
import requests

print("Milvus version:", pymilvus.__version__)
print("Sentence Transformers loaded:", sentence_transformers.__version__)

# 1. 配置设备和命令映射
devices = {
    "客厅灯": {"type": "light", "ip": "192.168.1.100", "status": "off"},
    "卧室灯": {"type": "light", "ip": "192.168.1.101", "status": "off"},
    "空调": {"type": "ac", "ip": "192.168.1.102", "status": "off", "temp": 26},
    "电视": {"type": "tv", "ip": "192.168.1.103", "status": "off", "volume": 20}
}

# 2. 初始化语音识别和合成引擎
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# 3. 初始化NLU模型
nlu = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# 4. 意图识别函数
def extract_intent(text):
    # 简化版意图识别
    intents = {
        "turn_on": ["打开", "开启", "启动"],
        "turn_off": ["关闭", "关掉", "停止"],
        "set_temp": ["设置温度", "调温度", "调到"],
        "query_status": ["状态", "怎么样", "是否"]
    }
    
    entities = []
    intent = None
    
    # 检查设备名称
    for device in devices.keys():
        if device in text:
            entities.append({"type": "device", "value": device})
    
    # 检查意图
    for intent_name, keywords in intents.items():
        for keyword in keywords:
            if keyword in text:
                intent = intent_name
                break
        if intent:
            break
    
    # 提取数值
    import re
    numbers = re.findall(r'\d+', text)
    if numbers and intent == "set_temp":
        entities.append({"type": "value", "value": int(numbers[0])})
    
    return {"intent": intent, "entities": entities}

# 5. 执行命令函数
def execute_command(intent, entities):
    if not intent or not entities:
        return "我没有理解您的指令"
    
    device = None
    value = None
    
    for entity in entities:
        if entity["type"] == "device":
            device = entity["value"]
        elif entity["type"] == "value":
            value = entity["value"]
    
    if not device or device not in devices:
        return "没有找到指定的设备"
    
    # 模拟设备控制
    if intent == "turn_on":
        devices[device]["status"] = "on"
        return f"已为您打开{device}"
    
    elif intent == "turn_off":
        devices[device]["status"] = "off"
        return f"已为您关闭{device}"
    
    elif intent == "set_temp" and devices[device]["type"] == "ac":
        if value:
            devices[device]["temp"] = value
            return f"已将{device}温度设置为{value}度"
        else:
            return "请指定温度值"
    
    elif intent == "query_status":
        status = devices[device]["status"]
        if status == "on":
            if devices[device]["type"] == "ac":
                return f"{device}已开启，当前温度{devices[device]['temp']}度"
            else:
                return f"{device}已开启"
        else:
            return f"{device}已关闭"
    
    return "我不明白您想做什么"

# 6. 主循环
def voice_control():
    print("智能家居语音助手已启动，说'你好助手'开始...")
    
    while True:
        # 监听唤醒词
        with sr.Microphone() as source:
            print("等待唤醒...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio, language="zh-CN")
            print(f"听到: {text}")
            
            if "你好助手" in text:
                engine.say("您好，请问有什么可以帮您?")
                engine.runAndWait()
                
                # 监听命令
                with sr.Microphone() as command_source:
                    print("请说出您的指令...")
                    recognizer.adjust_for_ambient_noise(command_source)
                    command_audio = recognizer.listen(command_source)
                
                command_text = recognizer.recognize_google(command_audio, language="zh-CN")
                print(f"指令: {command_text}")
                
                # 理解意图
                parsed = extract_intent(command_text)
                print(f"解析结果: {parsed}")
                
                # 执行命令
                response = execute_command(parsed["intent"], parsed["entities"])
                print(f"响应: {response}")
                
                engine.say(response)
                engine.runAndWait()
        
        except sr.UnknownValueError:
            print("无法识别语音")
        except sr.RequestError:
            print("无法连接到语音识别服务")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    voice_control()
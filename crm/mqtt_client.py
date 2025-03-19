import paho.mqtt.client as mqtt
import time
import json
import random

# MQTT服务器配置
MQTT_BROKER = "broker.emqx.io"  # 公共MQTT代理服务器
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60

# 主题配置
TOPIC_TEMPERATURE = "home/livingroom/temperature"
TOPIC_HUMIDITY = "home/livingroom/humidity"
TOPIC_COMMAND = "home/devices/+/command"  # 使用通配符订阅所有设备的命令

# 当连接到MQTT代理时的回调函数
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("已成功连接到MQTT代理")
        # 连接成功后订阅主题
        client.subscribe(TOPIC_COMMAND)
        print(f"已订阅主题: {TOPIC_COMMAND}")
    else:
        print(f"连接失败，返回码: {rc}")

# 当收到消息时的回调函数
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        print(f"收到消息 [{msg.topic}]: {payload}")
        
        # 如果是命令主题，解析并执行命令
        if "/command" in msg.topic:
            device_id = msg.topic.split("/")[2]
            try:
                command = json.loads(payload)
                process_command(device_id, command)
            except json.JSONDecodeError:
                print(f"无效的JSON命令: {payload}")
    except Exception as e:
        print(f"处理消息时出错: {e}")

# 处理命令
def process_command(device_id, command):
    if "action" in command:
        action = command["action"]
        print(f"执行设备 {device_id} 的 {action} 命令")
        
        # 这里可以添加实际的设备控制逻辑
        # 例如，如果是灯泡，可以控制开关
        if action == "on":
            print(f"打开设备 {device_id}")
        elif action == "off":
            print(f"关闭设备 {device_id}")
        
        # 发送状态更新
        status = {"status": action, "timestamp": time.time()}
        client.publish(f"home/devices/{device_id}/status", json.dumps(status))

# 创建MQTT客户端
def create_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    # 连接到MQTT代理
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        return client
    except Exception as e:
        print(f"连接MQTT代理时出错: {e}")
        return None

# 模拟发送传感器数据
def publish_sensor_data(client):
    while True:
        try:
            # 模拟温度数据
            temperature = round(random.uniform(20, 30), 1)
            client.publish(TOPIC_TEMPERATURE, str(temperature))
            print(f"已发布温度: {temperature}°C")
            
            # 模拟湿度数据
            humidity = round(random.uniform(40, 70), 1)
            client.publish(TOPIC_HUMIDITY, str(humidity))
            print(f"已发布湿度: {humidity}%")
            
            time.sleep(5)  # 每5秒发送一次数据
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发布数据时出错: {e}")
            time.sleep(5)  # 出错后等待5秒再重试

# 主函数
def main():
    client = create_mqtt_client()
    if client:
        # 启动网络循环
        client.loop_start()
        
        try:
            # 发布传感器数据
            publish_sensor_data(client)
        except KeyboardInterrupt:
            print("程序被用户中断")
        finally:
            # 断开连接
            client.loop_stop()
            client.disconnect()
            print("已断开MQTT连接")

if __name__ == "__main__":
    main() 
import paho.mqtt.client as mqtt
import json
import csv
import os
import time

MQTT_BROKER = "0eb00aca75ab4f56b36f74924b54b76b.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "test/topic"
MQTT_USER = "adrunio"
MQTT_PASSWORD = "Admin123"
CSV_FILE = "mqtt_data.csv"
MAX_LINES = 1000

line_count = 0
csv_file = None
csv_writer = None
client = None

def open_csv_file():
    global csv_file, csv_writer, line_count
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)  # 删除旧文件
    csv_file = open(CSV_FILE, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "x", "y", "z", "StateCode"])  # 写表头
    csv_file.flush()
    line_count = 0

def on_connect(client_obj, userdata, flags, rc):
    if rc == 0:
        print("连接MQTT成功")
        client_obj.subscribe(MQTT_TOPIC)
    else:
        print(f"连接失败，错误码：{rc}")

def on_message(client_obj, userdata, msg):
    global line_count, client, csv_file, csv_writer
    try:
        payload = msg.payload.decode('utf-8')
        print(f"收到消息: {payload}")

        data = json.loads(payload)
        time_str = data.get("time", "")
        x = data.get("x", "")
        y = data.get("y", "")
        z = data.get("z", "")
        state_code = data.get("StateCode", "")

        csv_writer.writerow([time_str, x, y, z, state_code])
        csv_file.flush()
        line_count += 1

        if line_count >= MAX_LINES:
            print(f"达到最大行数 {MAX_LINES}，断开连接，等待30秒")
            client.disconnect()
            csv_file.close()

            time.sleep(30)  # 等待30秒

            print("删除旧数据并重新开始写入")
            open_csv_file()

            print("重新连接MQTT")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_start()

    except Exception as e:
        print(f"处理消息出错: {e}")

def main():
    global client
    open_csv_file()

    client = mqtt.Client()
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.tls_set()  # 加密连接

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序终止")
    finally:
        if csv_file:
            csv_file.close()
        if client:
            client.disconnect()
            client.loop_stop()

if __name__ == "__main__":
    main()

from pymilvus import MilvusClient
import os 
import json
import random
from typing import Optional
import numpy as np
from uuid import uuid4
import shutil
import subprocess
import time

# 创建目录
if os.path.exists("test_milvus"):
    shutil.rmtree("test_milvus")
os.makedirs("test_milvus", exist_ok=True)

# 检查 Docker 是否运行
print("正在检查 Docker 状态...")
docker_status = subprocess.run(["docker", "info"], capture_output=True)
if docker_status.returncode != 0:
    print("Docker 未运行，请先启动 Docker")
    exit(1)

# 检查 Milvus 容器是否已存在
print("检查 Milvus 容器...")
check_container = subprocess.run(
    ["docker", "ps", "-a", "--filter", "name=milvus_standalone"], 
    capture_output=True, text=True
)

if "milvus_standalone" not in check_container.stdout:
    # 启动 Milvus 容器
    print("正在启动 Milvus 容器...")
    subprocess.run([
        "docker", "run", "-d", 
        "--name", "milvus_standalone", 
        "-p", "19530:19530", 
        "-p", "9091:9091", 
        "milvusdb/milvus:v2.3.3", "standalone"
    ])
else:
    # 检查容器是否运行
    check_running = subprocess.run(
        ["docker", "ps", "--filter", "name=milvus_standalone"], 
        capture_output=True, text=True
    )
    if "milvus_standalone" not in check_running.stdout:
        print("Milvus 容器已存在但未运行，正在启动...")
        subprocess.run(["docker", "start", "milvus_standalone"])
    else:
        print("Milvus 容器已在运行")

# 等待服务器启动
print("等待 Milvus 服务器启动...")
time.sleep(10)  # 等待10秒

# 使用标准连接方式，而不是本地文件路径
client = MilvusClient(uri="http://localhost:19530")

TABLE_NAME = "textsearch4testv2"
DIM_VALUE = 10

# 如果集合已存在，先删除
if client.has_collection(TABLE_NAME):
    client.drop_collection(TABLE_NAME)

# 创建集合
client.create_collection(
    collection_name=TABLE_NAME,
    dimension=DIM_VALUE,
)

print(f"集合 {TABLE_NAME} 创建成功！")

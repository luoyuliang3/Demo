from pymilvus import MilvusClient
import numpy as np
from sentence_transformers import SentenceTransformer

# 初始化文本向量模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 连接到Milvus服务器
client = MilvusClient(uri="http://localhost:19530")

# 集合名称
COLLECTION_NAME = "text_search_demo"
VECTOR_DIM = 384  # 向量维度，取决于模型

# 创建集合
def setup_collection():
    # 如果集合已存在，先删除
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    # 创建新集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=VECTOR_DIM,
        primary_field_name="id",
        vector_field_name="embedding"
    )
    print(f"集合 {COLLECTION_NAME} 创建成功")

# 将文本转换为向量
def text_to_vector(text):
    return model.encode(text).tolist()

# 插入数据
def insert_data(texts):
    entities = []
    for i, text in enumerate(texts):
        entities.append({
            "id": i,
            "text": text,
            "embedding": text_to_vector(text)
        })
    
    client.insert(
        collection_name=COLLECTION_NAME,
        data=entities
    )
    print(f"已插入 {len(texts)} 条数据")

# 创建索引
def create_index():
    client.create_index(
        collection_name=COLLECTION_NAME,
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )
    print("索引创建成功")
    
    # 加载集合到内存
    client.load_collection(COLLECTION_NAME)

# 搜索相似文本
def search_similar(query_text, limit=5):
    query_vector = text_to_vector(query_text)
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        field_name="embedding",
        limit=limit,
        output_fields=["text"]
    )
    
    print(f"查询: '{query_text}'")
    print("搜索结果:")
    for i, hit in enumerate(results[0]):
        print(f"{i+1}. 相似度: {1-hit['distance']:.4f}, 文本: {hit['entity']['text']}")

# 主函数
def main():
    # 示例文本数据
    texts = [
        "智能家居可以提高生活品质",
        "物联网技术正在改变我们的生活方式",
        "人工智能在家庭自动化中的应用",
        "语音控制是智能家居的重要功能",
        "智能照明系统可以节省能源",
        "家庭安防系统提供全天候保护",
        "智能温控系统可以优化室内温度",
        "远程控制家电带来便利",
        "智能门锁提高了家庭安全性",
        "智能音箱是智能家居的中心控制器"
    ]
    
    # 设置集合
    setup_collection()
    
    # 插入数据
    insert_data(texts)
    
    # 创建索引
    create_index()
    
    # 搜索示例
    search_similar("如何用AI控制家里的设备")
    print("\n")
    search_similar("家庭安全系统")

if __name__ == "__main__":
    main() 
from pymilvus import MilvusClient
import numpy as np
import random

# 创建 Milvus 客户端
# 如果使用默认设置的本地 Milvus 服务，可以不传参数
# 如果连接远程服务器，需要指定 uri
client = MilvusClient(uri="http://localhost:19530")

# 定义集合名称
collection_name = "example_collection"

# 删除已存在的同名集合（如果存在）
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 创建集合
# 定义向量维度
dim = 128

# 创建集合，指定主键字段和向量字段
client.create_collection(
    collection_name=collection_name,
    dimension=dim,
    primary_field_name="id",
    vector_field_name="embedding"
)

# 准备要插入的数据
# 生成随机向量数据
num_entities = 1000
entities = [
    {
        "id": i,  # 主键
        "embedding": np.random.random(dim).tolist(),  # 向量数据
        "text": f"这是第 {i} 个文档",  # 额外的文本字段
        "score": random.uniform(0, 100)  # 额外的数值字段
    }
    for i in range(num_entities)
]

# 插入数据
client.insert(
    collection_name=collection_name,
    data=entities
)

# 创建索引以加速搜索
client.create_index(
    collection_name=collection_name,
    field_name="embedding",
    index_type="IVF_FLAT",  # 索引类型
    metric_type="L2",       # 距离度量方式
    params={"nlist": 128}   # 索引参数
)

# 加载集合到内存
client.load_collection(collection_name)

# 执行向量搜索
# 生成一个随机查询向量
query_vector = np.random.random(dim).tolist()

# 执行搜索
search_results = client.search(
    collection_name=collection_name,
    data=[query_vector],    # 查询向量
    field_name="embedding", # 要搜索的向量字段
    limit=5,                # 返回最相似的5个结果
    output_fields=["text", "score"]  # 返回这些额外字段
)

# 打印搜索结果
print("搜索结果:")
for i, result in enumerate(search_results):
    print(f"\n查询向量 {i} 的结果:")
    for hit in result:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}, 文本: {hit['entity']['text']}, 分数: {hit['entity']['score']}")

# 按条件过滤搜索
filtered_results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    field_name="embedding",
    limit=5,
    filter="score > 50",  # 只返回分数大于50的结果
    output_fields=["text", "score"]
)

# 打印过滤后的搜索结果
print("\n过滤后的搜索结果 (score > 50):")
for i, result in enumerate(filtered_results):
    print(f"\n查询向量 {i} 的结果:")
    for hit in result:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}, 文本: {hit['entity']['text']}, 分数: {hit['entity']['score']}")

# 执行混合查询（向量 + 标量查询）
hybrid_results = client.query(
    collection_name=collection_name,
    filter="score > 70",
    output_fields=["id", "score", "text"],
    limit=5
)

# 打印混合查询结果
print("\n混合查询结果 (score > 70):")
for result in hybrid_results:
    print(f"ID: {result['id']}, 文本: {result['text']}, 分数: {result['score']}")

# 清理：释放集合并删除
client.release_collection(collection_name)
client.drop_collection(collection_name)

print("\n示例完成，集合已删除")
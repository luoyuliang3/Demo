import os
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import requests
import json
import time

# 初始化文本向量模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 连接到Milvus服务器
client = MilvusClient(uri="http://localhost:19530")

# 集合名称和向量维度
COLLECTION_NAME = "smart_home_knowledge"
VECTOR_DIM = 384  # 向量维度，取决于模型

# 1. 创建集合
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

# 2. 文本向量化
def text_to_vector(text):
    return model.encode(text).tolist()

# 3. 文档处理和加载
def process_documents(documents):
    """处理文档并分割成段落"""
    chunks = []
    for doc in documents:
        # 简单按段落分割
        paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
    return chunks

# 4. 向量存储
def store_documents(chunks):
    entities = []
    for i, chunk in enumerate(chunks):
        entities.append({
            "id": i,
            "text": chunk,
            "embedding": text_to_vector(chunk)
        })
    
    client.insert(
        collection_name=COLLECTION_NAME,
        data=entities
    )
    print(f"已插入 {len(chunks)} 个文档块")

# 5. 创建索引
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

# 6. 检索相关文档
def retrieve_relevant_docs(query, top_k=3):
    query_vector = text_to_vector(query)
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        field_name="embedding",
        limit=top_k,
        output_fields=["text"]
    )
    
    relevant_docs = []
    for hit in results[0]:
        relevant_docs.append({
            "text": hit['entity']['text'],
            "score": 1 - hit['distance']  # 转换距离为相似度分数
        })
    
    return relevant_docs

# 7. 调用LLM生成回答
def generate_answer(query, relevant_docs):
    """使用检索到的文档增强LLM回答"""
    # 构建提示
    context = "\n".join([f"文档 {i+1}: {doc['text']}" for i, doc in enumerate(relevant_docs)])
    
    prompt = f"""请基于以下智能家居领域的文档回答用户的问题。
如果文档中没有相关信息，请诚实地说你不知道。

文档内容:
{context}

用户问题: {query}

回答:"""

    # 这里使用本地模型或API调用LLM
    # 示例使用模拟的回答生成
    answer = simulate_llm_response(prompt, query, relevant_docs)
    return answer

# 模拟LLM响应（实际应用中应替换为真实的LLM API调用）
def simulate_llm_response(prompt, query, docs):
    """模拟LLM响应，实际应用中应替换为真实API调用"""
    if not docs:
        return "抱歉，我没有找到相关的信息来回答您的问题。"
    
    # 简单模拟，实际应用中应使用真实LLM
    if "智能家居" in query or "智能" in query:
        return f"根据我检索到的信息，{docs[0]['text']} 此外，{docs[1]['text'] if len(docs) > 1 else ''}"
    elif "安全" in query:
        return f"关于智能家居安全，文档提到：{docs[0]['text']}"
    else:
        return f"您询问的是关于{query}的问题。根据我检索到的资料：{docs[0]['text']}"

# 8. RAG系统主流程
def rag_system(query):
    print(f"用户问题: {query}")
    
    # 检索相关文档
    start_time = time.time()
    relevant_docs = retrieve_relevant_docs(query)
    retrieval_time = time.time() - start_time
    
    print(f"检索到 {len(relevant_docs)} 个相关文档 (耗时: {retrieval_time:.2f}秒)")
    for i, doc in enumerate(relevant_docs):
        print(f"文档 {i+1} (相似度: {doc['score']:.4f}): {doc['text'][:100]}...")
    
    # 生成回答
    start_time = time.time()
    answer = generate_answer(query, relevant_docs)
    generation_time = time.time() - start_time
    
    print(f"\n回答 (生成耗时: {generation_time:.2f}秒):")
    print(answer)
    
    return {
        "query": query,
        "relevant_docs": relevant_docs,
        "answer": answer,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time
    }

# 9. 主函数
def main():
    # 示例智能家居领域知识
    documents = [
        """
        智能家居系统是指通过各种传感器、控制器和智能设备，实现家庭环境的自动化控制和管理的系统。
        智能家居可以提高生活品质，节省能源，增强安全性，并为居住者提供更多便利。
        现代智能家居系统通常包括照明控制、温度调节、安防监控、娱乐系统和家电控制等功能。
        """,
        
        """
        智能照明系统允许用户通过手机应用、语音命令或自动化场景控制家中的灯光。
        用户可以调整亮度、色温，甚至改变灯光颜色，创造不同的氛围。
        智能照明还可以根据时间、日照或人员存在自动调整，有效节省能源。
        常见的智能照明品牌包括飞利浦Hue、小米智能灯和宜家TRÅDFRI系列。
        """,
        
        """
        智能安防系统是智能家居的重要组成部分，包括智能门锁、监控摄像头、门窗传感器和动作探测器等。
        这些设备可以实时监控家庭安全状况，并在检测到异常时立即通知用户。
        智能门锁支持指纹识别、密码输入、NFC卡片和远程解锁等多种开门方式，提高了家庭安全性。
        高级安防系统还可以与专业安保服务连接，提供24小时监控和紧急响应。
        """,
        
        """
        智能温控系统通过智能恒温器控制家中的暖气、空调和通风系统，保持舒适的室内温度。
        这些系统可以学习用户习惯，自动调整温度设置，最大化舒适度和能源效率。
        智能恒温器可以通过地理围栏技术检测用户是否在家，自动调整温度设置。
        研究表明，智能温控系统平均可以节省10-15%的能源消耗。
        """,
        
        """
        语音控制是现代智能家居系统的核心功能，通过智能音箱或语音助手实现。
        用户可以通过简单的语音命令控制灯光、温度、音乐、电视和其他智能设备。
        主流的语音助手包括小爱同学、天猫精灵、小度、Alexa和Google Assistant等。
        语音控制系统通常支持自然语言处理，能够理解复杂的命令和上下文信息。
        """,
        
        """
        智能家居设备的互操作性是行业面临的主要挑战之一。
        不同品牌和系统之间的兼容性问题可能导致用户体验不佳。
        Matter协议是一项新的智能家居标准，旨在解决设备互操作性问题，让不同品牌的设备能够无缝协作。
        其他常见的智能家居协议包括Zigbee、Z-Wave、Wi-Fi和蓝牙等。
        """,
        
        """
        智能家居系统的隐私和安全问题日益受到关注。
        联网设备可能存在安全漏洞，导致个人数据泄露或设备被黑客控制。
        用户应定期更新设备固件，使用强密码，并了解设备收集的数据类型。
        一些高端智能家居系统提供端到端加密和本地处理功能，减少数据传输和云存储的风险。
        """
    ]
    
    # 设置集合
    setup_collection()
    
    # 处理文档
    chunks = process_documents(documents)
    
    # 存储文档
    store_documents(chunks)
    
    # 创建索引
    create_index()
    
    # 测试RAG系统
    test_queries = [
        "智能家居系统有哪些主要功能？",
        "智能照明系统如何节省能源？",
        "智能家居的安全问题有哪些？",
        "Matter协议是什么？",
        "如何通过语音控制智能家居设备？"
    ]
    
    for query in test_queries:
        result = rag_system(query)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main() 
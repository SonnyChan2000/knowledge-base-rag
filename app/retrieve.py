# app/retrieve.py 项目一: 智能知识库系统 检索模块
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "db"

def get_vectorstore():
    """打开上次入库的向量库(只读已有库, 不重复建)"""
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device":"cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        embedding_function=embedding,
        persist_directory=str(DB_DIR)
    )

def search(query: str, top_k: int = 3):
    """按问题检索最相关的top_k个向量块, 返回(内容, 来源, 相似度)"""
    store = get_vectorstore()
    results = store.similarity_search_with_score(query, k=top_k)
    print(f"\n问题: {query}\n{'='*50}")
    for i, (doc, score) in enumerate(results, 1):
        # score是余弦距离, 越小越相关
        source = doc.metadata.get("source", "未知")
        snippet = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
        print(f"[{i}] 来源: {source} | 相似度: {score:.4f}")
        print(f"      内容: {snippet}\n")
    return results

if __name__ == "__main__":
    search("游戏报错有哪些解决方法")
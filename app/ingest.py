# app/ingest.py
# 项目一: 智能知识库系统 - 文档加载与入库模块
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 项目的data和db目录(相对项目根定位)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

# ------------- 练习1 文档加载器 ------------------
def load_documents(folder: Path = DATA_DIR):
    """遍历data/下所有文件, 按扩展名解析, 返回带source元数据的Document列表"""
    docs = []
    for file in folder.iterdir():
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif suffix == ".txt" or suffix ==  "md":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            print(f"跳过不支持的类型: {file.name}")
            continue
        # 解析出的每段都打上source标记(后面引用来源靠它)
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = file.name
        docs.extend(loaded)
        print(f"已加载{file.name} -> {len(loaded)}段")
    return docs

# -------------  练习2 分割 + 向量化 + 入库 ---------
def ingest_folder(folder: Path = DATA_DIR):
    """把 data/下文档分割成块、向量化后存入Chroma"""
    # 1.加载
    docs = load_documents(folder)
    if not docs:
        print("没有可入库的文档")
        return

    # 2.分割(中文一起按段落和标点切)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "!", "?", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"分割完成: {len(docs)}段 -> {len(chunks)}块")

    # 3.向量化 + 存库
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(DB_DIR)
    )
    # 打印来源分布, 验证元数据没丟
    from collections import Counter
    sources = Counter(c.metadata.get("source") for c in chunks)
    print("来源分布: ", dict(sources))
    print(f"入库完成, 共{len(chunks)}块, 存于{DB_DIR}")

# ------------------- 入口 -----------------
if __name__ == "__main__":
    ingest_folder()
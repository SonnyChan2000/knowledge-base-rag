# 项目一: 智能知识库系统 问答模块(检索 + 生成)
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "db"

# 1. 连接向量库(复用retrieve的思路)
def get_vectorstore():
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        embedding_function=embedding,
        persist_directory=str(DB_DIR),
    )

# 2.组装问答链
def build_chain():
    store = get_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库问答助手。请只根据下面的资料回答问题；资料里没有的内容, 明确说'资料中没有相关信息', 不要编造。\n\n 资料:\n{context}"),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )

    # 手动串起来: 取资料 -> 拼prompt -> 让llm生成
    def answer(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        messages = prompt.format_messages(context=context, question=question)
        reply = llm.invoke(messages)
        return reply.content, docs

    return answer

if __name__ == "__main__":
    answer = build_chain()
    while True:
        q = input("\n请输入问题(输入exit退出): ")
        if q.lower() == "exit":
            break
        reply, docs = answer(q)
        print(f"\n 回答: {reply}")
        print("----引用来源----")
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "未知")
            if src not in seen:
                seen.add(src)
                print(f"  · {src}")

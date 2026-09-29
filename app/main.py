# 项目一.智能知识库系统 FastAPI服务层(含多轮记忆)
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"

# 1.组件: 向量库/提示词/LLM(只组装一次)
embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Chroma(embedding_function=embedding, persist_directory=str(DB_DIR))
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库问答助手。请只根据下面的资料回答；资料里没有的内容, 就说'资料中没有相关信息', 不要编造。\n\n资料: \n{context}"),
    ("human", "对话历史: \n{history}\n\n当前问题: {question}")
])

llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4"    
)

# 2.多轮记忆: thread_id -> 历史列表
sessions = {}

def get_history(thread_id: str) -> str:
    """把某个会话的历史拼接成文本"""
    lines = []
    for role, text in sessions.get(thread_id, []):
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

def add_history(thread_id:str, user_q:str, ai_a:str):
    """往某个会话追加一轮对话"""
    s = sessions.setdefault(thread_id, [])
    s.append(("用户", user_q))
    s.append(("AI", ai_a))
    if len(s) > 10:
        del s[:2]

# 3.真正回答问题
def answer(thread_id:str, question:str):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    history = get_history(thread_id)
    messages = prompt.format_messages(context=context, history=history, question=question)
    reply = llm.invoke(messages).content
    add_history(thread_id, question, reply)
    sources = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "未知")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return {"answer": reply, "sources": sources}

# 4. FastAPI应用
app = FastAPI(title="智能知识库系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    thread_id:str = "default"
    question: str

@app.post("/ask")
def ask(q: Question):
    result = answer(q.thread_id, q.question)
    return result

@app.get("/")
def home():
    return {"msg": "知识库服务已启动", "demo": "POST/ask 传入{thread_id, question}"}

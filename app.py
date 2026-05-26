import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from pypdf import PdfReader
from docx import Document
import os
import hashlib

load_dotenv()

st.set_page_config(page_title="AI文档助手", page_icon="📚", layout="wide")
st.title("📚 AI 文档问答助手（FAISS版）")
st.caption("上传你的文档，直接用中文提问！")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None

# ========== 读取文档 ==========
def read_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        return '\n'.join([p.extract_text() or '' for p in reader.pages])
    elif name.endswith('.docx'):
        doc = Document(uploaded_file)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    return ""

# ========== 缓存Embedding模型 ==========
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

# ========== 用哈希值缓存向量库 ==========
@st.cache_resource
def build_vectorstore(doc_hash: str, text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, get_embeddings())

def get_vectorstore():
    h = st.session_state.get("doc_hash")
    t = st.session_state.get("doc_text")
    if not h or not t:
        return None
    return build_vectorstore(h, t)

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("📁 上传文档")
    uploaded_file = st.file_uploader(
        "支持 PDF / Word / TXT",
        type=["pdf", "docx", "txt"]
    )
    if uploaded_file:
        with st.spinner("正在读取并建立索引..."):
            text = read_file(uploaded_file)
            if text.strip():
                doc_hash = hashlib.md5(text.encode()).hexdigest()
                st.session_state.doc_hash = doc_hash
                st.session_state.doc_text = text
                vs = build_vectorstore(doc_hash, text)
                import store
                store.set_vs(vs)
                st.success(f"✅ 文档加载成功！共 {len(text)} 字符")
            else:
                st.error("文档内容为空")

    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ========== LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

# ========== 工具 ==========
@tool
def search_document(query: str) -> str:
    """在已加载的文档中搜索相关内容，输入问题关键词"""
    import store
    vs = store.get_vs()
    if vs is None:
        return "还没有加载文档，请先在左侧上传文件！"
    docs = vs.similarity_search(query, k=3)
    return '\n---\n'.join([d.page_content for d in docs])

SYSTEM_PROMPT = """你是一个专业的文档助手。
规则：
1. 用户问任何问题，必须先调用 search_document 搜索文档
2. 根据搜索结果用中文详细回答
3. 文档里找不到才说"文档中未找到相关信息"
"""

agent = create_react_agent(llm, [search_document])

# ========== 显示历史消息 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========== 输入框 ==========
if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = agent.invoke({
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    {"role": "user", "content": prompt}
                ]
            })
            answer = result["messages"][-1].content
        st.write(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
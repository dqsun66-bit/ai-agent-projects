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

load_dotenv()

def read_file(file_path: str) -> str:
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf':
        reader = PdfReader(file_path)
        return '\n'.join([page.extract_text() for page in reader.pages])
    elif ext == 'docx':
        doc = Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "不支持的文件格式"

vectorstore = None

def build_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

@tool
def load_document(file_path: str) -> str:
    """加载文档到记忆中，输入文件完整路径，支持PDF/Word/TXT"""
    global vectorstore
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"
    text = read_file(file_path)
    if not text.strip():
        return "文件内容为空"
    vectorstore = build_vectorstore(text)
    return f"文档加载成功！共读取 {len(text)} 个字符，可以开始提问了！"

@tool
def search_document(query: str) -> str:
    """在已加载的文档中搜索相关内容，输入问题关键词"""
    global vectorstore
    if vectorstore is None:
        return "还没有加载文档，请先告诉我文件路径"
    docs = vectorstore.similarity_search(query, k=3)
    return '\n---\n'.join([d.page_content for d in docs])

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

SYSTEM_PROMPT = """你是一个专业的文档助手，已加载了用户的个人文档。
规则：
1. 用户问任何问题，必须先调用 search_document 工具搜索文档
2. 根据搜索结果回答，简洁准确，用中文
3. 文档里找不到才说"文档中未找到相关信息"
"""

tools = [load_document, search_document]
agent = create_react_agent(llm, tools)

print("📚 RAG文档助手启动！")
print("使用方法：告诉我文件路径，我来读取并回答问题\n")

while True:
    q = input("你: ")
    if q.lower() == "quit":
        break
    result = agent.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            {"role": "user", "content": q}
        ]
    })
    print(f"\nAgent: {result['messages'][-1].content}\n")
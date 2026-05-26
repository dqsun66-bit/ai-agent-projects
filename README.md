# 🤖 AI Agent 项目集

基于 LangChain + LangGraph + DeepSeek 构建的 AI Agent 系统

## 项目列表

### 1. 多工具 AI Agent（agent.py）
- 自然语言对话
- 数学计算工具
- 天气查询工具

### 2. RAG 文档问答系统（rag.py）
- 支持 PDF / Word / TXT 文档
- 本地向量数据库（FAISS）
- 自动搜索文档回答问题

## 技术栈
- Python 3.11
- LangChain / LangGraph
- DeepSeek API
- FAISS 向量数据库
- HuggingFace Embeddings

## 快速开始

1. 安装依赖
pip install langchain langchain-openai langgraph faiss-cpu pypdf python-docx sentence-transformers python-dotenv

2. 配置 API Key，创建 .env 文件
OPENAI_API_KEY=你的DeepSeek Key
OPENAI_BASE_URL=https://api.deepseek.com

3. 运行 Agent
python agent.py

4. 运行 RAG 系统
python rag.py

## 作者
正在学习 AI Agent 开发，目标成为 AI 应用工程师

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import os

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如：2+2 或 100*365"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "计算失败，请检查表达式"

@tool
def get_weather(city: str) -> str:
    """查询城市天气，输入城市名称"""
    return f"{city}今天天气晴朗，温度25°C，适合出行！"

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL")
)

tools = [calculator, get_weather]
agent = create_react_agent(llm, tools)

print("🤖 Agent启动成功！输入问题开始对话，输入quit退出\n")
while True:
    question = input("你: ")
    if question.lower() == "quit":
        break
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\nAgent: {result['messages'][-1].content}\n")
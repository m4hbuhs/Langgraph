from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from typing import Literal,TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
class Chatstate(TypedDict):
    messages :Annotated[list[BaseMessage],add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

def chat_node(state : Chatstate) ->Chatstate:
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(Chatstate)

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

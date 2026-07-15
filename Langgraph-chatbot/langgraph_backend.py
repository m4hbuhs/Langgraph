from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from typing import Literal,TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
class Chatstate(TypedDict):
    messages :Annotated[list[BaseMessage],add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

def chat_node(state : Chatstate) ->Chatstate:
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}
checkpointer = MemorySaver()
graph = StateGraph(Chatstate)

graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

for message_chunk,metadata in chatbot.stream(
    {'messages':[HumanMessage(content='what is the recipe to make pasta')]},
    config={'configurable':{'thread_id':'thread-1'}},
    stream_mode="messages"
):
    if(message_chunk.content):
        print(message_chunk.content,end=" ",flush=True)
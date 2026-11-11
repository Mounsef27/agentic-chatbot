
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://localhost:11434",
    temperature=0,
)

# Define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chatbot node
def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# Initialize checkpointer
checkpointer = InMemorySaver()

# Create graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)


# Run chatbot
if __name__ == "__main__":

    while True:

        message = input("Write your prompt here: ")

        if message.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        config = {
            "configurable": {
                "thread_id": "thread-1"
            }
        }

        response = chatbot.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        print("AI:", response["messages"][-1].content)

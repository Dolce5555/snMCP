from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from ruamel.yaml import YAML
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator
from IPython.display import Image, display
from pydantic import BaseModel, Field

## secret情報の取得
filePath = "/app/backend/secrets.yaml"
yaml = YAML(typ = "safe", pure = True)
with open(filePath, "r", encoding="utf-8") as f:
    secrets = yaml.load(f)
# llmKind = "aiStudio"
llmKind = "baseLlm"
## LLMの初期化
model = init_chat_model(
    model = secrets[llmKind]["modelName"],
    base_url = secrets[llmKind]["baseUrl"],
    model_provider = secrets[llmKind]["modelProvider"],
    api_key = secrets[llmKind]["apiKey"],
)

class searchInput(BaseModel):
    """
    検索に必要なインプット情報の定義
    """
    query: str = Field(
        description="検索するためのクエリ"
    )


def weather(query: str) -> str:
    """
    地名を受け取り、その場所の天気を検索するツール

    Args:
        query: 天気を検索するためのクエリ
    Returns:
        検索結果のテキスト
    """
    return "晴れ"



# class searchInput(BaseModel):
#     """インプット情報の定義"""
#     a: int = Field(
#         description="First int"
#     )
#     b: int = Field(
#         description="Second int"
#     )

# # Define tools
# @tool("multiply", args_schema=searchInput, return_direct=True, description = "Return the product of 2 integers `a` and `b`.")
# def multiply(a: int, b: int) -> int:
#     """Multiply `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a * b


# @tool("add", args_schema=searchInput, return_direct=True)
# def add(a: int, b: int) -> int:
#     """Adds `a` and `b`.

#     Args:
#         a: First int
#         b: Second int
#     """
#     return a + b


# # Augment the LLM with tools
# tools = [add, multiply]
# tools_by_name = {tool.name: tool for tool in tools}
# model_with_tools = model.bind_tools(tools)

# # Step 2: Define state

# from langchain.messages import AnyMessage
# from typing_extensions import TypedDict, Annotated
# import operator


# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
#     llm_calls: int

# # Step 3: Define model node
# from langchain.messages import SystemMessage


# def llm_call(state: dict):
#     """LLM decides whether to call a tool or not"""

#     messages = [
#         SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
#         ] + state["messages"]
#     print(f"*****\n{messages}\n*****")
#     response = {
#         "messages": [
#             model_with_tools.invoke(
#                 [
#                     SystemMessage(
#                         content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
#                     )
#                 ]
#                 + state["messages"]
#             )
#         ],
#         "llm_calls": state.get('llm_calls', 0) + 1
#     }
#     return response


# # Step 4: Define tool node

# from langchain.messages import ToolMessage


# def tool_node(state: dict):
#     """Performs the tool call"""

#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         observation = tool.invoke(tool_call["args"])
#         result.append(ToolMessage(content=f"tool {tool.name} call result: {observation}", tool_call_id=tool_call["id"]))
#     return {"messages": result}

# # Step 5: Define logic to determine whether to end

# from typing import Literal
# from langgraph.graph import StateGraph, START, END


# # Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
# def should_continue(state: MessagesState) -> Literal["tool_node", END]:
#     """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#     messages = state["messages"]
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then perform an action
#     if last_message.tool_calls:
#         return "tool_node"

#     # Otherwise, we stop (reply to the user)
#     return END

# # Step 6: Build agent

# # Build workflow
# agent_builder = StateGraph(MessagesState)

# # Add nodes
# agent_builder.add_node("llm_call", llm_call)
# agent_builder.add_node("tool_node", tool_node)

# # Add edges to connect nodes
# agent_builder.add_edge(START, "llm_call")
# agent_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     ["tool_node", END]
# )
# agent_builder.add_edge("tool_node", "llm_call")

# # Compile the agent
# agent = agent_builder.compile()


# # from IPython.display import Image, display
# # # Show the agent
# # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# # Invoke
# from langchain.messages import HumanMessage
# messages = [HumanMessage(content="11足す14足す15の答えを教えて。")]
# messages = agent.invoke({"messages": messages})
# for m in messages["messages"]:
#     m.pretty_print()

# # call our graph with streaming to see the steps
# # for state in agent.stream(messages, stream_mode="values"):
# #     last_message = state["messages"][-1]
# #     last_message.pretty_print()
from langchain_core.tools import StructuredTool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from ruamel.yaml import YAML
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator
from IPython.display import Image, display
from pydantic import BaseModel, Field

"""
OpenAI APIのMessagesの構造

"""

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

class StateMethod(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

class snAgent():
    def __init__(self):
        # tool を使えるように model を定義
        self.model = model
        
        # selfがバインドされたメソッドをToolとして登録する
        weather_tool = StructuredTool.from_function(
            func=self.weather,
            name="weather",
            description="{query} の天気を回答する",
            args_schema=searchInput
        )
        
        tools = [weather_tool]
        ## toolの名前でtoolを取得できるようにする {"tool_name": tool_instance, ...}
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.model_with_tools = model.bind_tools(tools)
        # LangGraphのagentを定義
        self.agent_builder = StateGraph(StateMethod)
        self.agent_builder.add_node("llm_call", self.llm_call)
        self.agent_builder.add_node("tool_node", self.tool_node)
        self.agent_builder.add_edge(START, "llm_call")
        self.agent_builder.add_conditional_edges(
            "llm_call",
            self.should_continue,
            ["tool_node", END]# should_continueがtool_nodeかENDを返す
        )
        self.agent_builder.add_edge("tool_node", "llm_call")
        self.agent = self.agent_builder.compile()
        self.agent.get_graph(xray=True).draw_mermaid_png(output_file_path = "/app/backend/img/test_workflow.png")


    def weather(self, query: str):
        """
        {query} の天気を回答する
        Args:
            query (str): 地名
        """
        print(f" *****{query}*****")
        return f"{query}の天気は晴れです"


    def llm_call(self, state: dict):
        """
        LLMを呼び出して、ツールを使用するかどうかを判断する
        Return:
            StateMethod:
                messages: LLMからの応答
                llm_calls: LLMの呼び出し回数
        """
        print(f" ***** llm call *****\n{state}\n")

        messages = [
            SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
            ] + state["messages"]
        response = {
            "messages": [
                self.model_with_tools.invoke(
                    # [
                    #     SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
                    # ]
                    #  + state["messages"]
                    messages
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }
        return response
    

    def tool_node(self, state: dict):
        """
        ツールを使用する
        Return:
            StateMethod:
                messages: ツールの実行結果
                llm_calls: LLMの呼び出し回数 (LLM を呼び出さないため引き継ぎ)
        """
        print(f" ***** tool call *****\n{state}\n")
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            result.append(
                ToolMessage(
                    content=tool.invoke(tool_call["args"]),
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": result, "llm_calls": state["llm_calls"]}


    def should_continue(self, state: dict):
        """
        ツールを使用するかどうかを判断する
        Return:
            str: "tool_node"
               or
            END: END
        """
        print(f" ***** ツール使用か否か判断 *****\n")
        ##essagesの最後の要素がtool_callsを持っているか判断する
        if state["messages"][-1].tool_calls:
            return "tool_node"
        else:
            return END


    def run(self, input: str):
        """
        Agent を実行する
        Args:
            input (str): ユーザーの入力
        """
        result = self.agent.invoke({"messages": [HumanMessage(content=input)]})
        print(result)
        return result


if __name__ == "__main__":
    snAgent = snAgent()
    snAgent.run("こんにちは。今日の東京の天気は？")
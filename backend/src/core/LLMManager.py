import logging
from langchain.chat_models import init_chat_model
from openai import OpenAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, Union
import json

####################

## プロンプトテンプレートを型規定
class ChatTemplate(BaseModel):
    sys_prompt: str
    usr_prompt: str
    ai_prompt: Optional[str] = None
    old_messages: Optional[list] = None

    @property
    def mk_messages(self):
        messages = [SystemMessage(content=self.sys_prompt)]
        if self.old_messages:
            messages.extend(self.old_messages)
        messages.append(HumanMessage(content=self.usr_prompt))
        return messages

####################

## LLMへの接続を抽象化
class LLMManager:
    def __init__(self, supplier: str, secrets, logger: logging.Logger = None, logLevel: int = logging.INFO, llmKind = "baseLlm"):
        """
        supplier: "vllm" or 新規 model サプライヤ (openai, vertexai, etc.) 
        secrets: secrets.yaml
        logger: logger
        logLevel: log level
        """
        self.supplier = supplier
        self.secrets = secrets
        self.logger = logger.getChild(self.__class__.__name__) or self.getLogger(f"{__file__.split("/")[-1]}_{self.__class__.__name__}", logLevel)
        self.logLevel = logLevel
        if self.supplier == "vllm":
            self.manager = VLLMManager(secrets, self.logger, llmKind)
        elif self.supplier == "openai":
            self.manager = OpenAIManager(secrets, self.logger, llmKind)
        else:
            self.logger.error(f"不明なサプライヤが指定されました: {self.supplier}")
            raise ValueError(f"不明なサプライヤが指定されました: {self.supplier}")

    def getLogger(self, name: str, logLevel: int = logging.INFO) -> logging.Logger:
        """
        シンプルなロガー初期化ヘルパー。
        アプリ側でハンドラを既に設定している場合は干渉しない。
        """        
        name = f"{__file__.split("/")[-1]}_{self.__class__.__name__}"
        logger = logging.getLogger(name)
        logger.setLevel(logLevel)
        # 既にハンドラが無ければデフォルトの StreamHandler を追加
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logLevel)
            fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    def invoke(self, prompt: Union[str, ChatTemplate, list]):
        """LLM呼び出しを内部マネージャーに委譲する"""
        return self.manager.invoke(prompt)

    def bind_tools(self, tools: list):
        """ツールバインドを内部マネージャーに委譲する"""
        self.manager.bind_tools(tools)

####################

## 自分のつかうモデルサプライヤを抽象化
class ModelBaseManager(ABC):
    @abstractmethod
    def __init__(self, secrets, logger: logging.Logger, llmKind: str):
        pass
    @abstractmethod
    def invoke(self, prompt: Union[str, ChatTemplate, list]):
        pass
    @abstractmethod
    def bind_tools(self, tools: list):
        pass
    @abstractmethod
    def stockMessages(self, response: list):
        pass

## VLLM を使ったモデルサプライヤ
class VLLMManager(ModelBaseManager):
    def __init__(self, secrets, logger: logging.Logger, llmKind: str):
        """
        """
        self.logger = logger
        self.model = init_chat_model(
            model = secrets[llmKind]["modelName"],
            base_url = secrets[llmKind]["baseUrl"],
            model_provider = "openai",
            api_key = secrets[llmKind]["apiKey"],
        )
        # self.embed_model = OpenAIEmbeddings(
        #     model = secrets["embeddingModel"]["modelName"],
        #     base_url = secrets["embeddingModel"]["baseUrl"],
        #     api_key = secrets["embeddingModel"]["apiKey"],
        # )
        self.sys_prompt = secrets[llmKind]["sys_prompt"]

    def invoke(self, prompt: Union[str, ChatTemplate, list]):
        """
        promptを受け取り、LLMに投げかける。
        文字列(usr_prompt)、ChatTemplateインスタンス、あるいは整形済みのメッセージリストを受け入れる。
        """
        if isinstance(prompt, str):
            messages = ChatTemplate(
                sys_prompt=self.sys_prompt,
                usr_prompt=prompt,
            ).mk_messages
        elif isinstance(prompt, ChatTemplate):
            messages = prompt.mk_messages
        elif isinstance(prompt, list):
            messages = prompt
        else:
            self.logger.error(f"不正なプロンプト形式です: {type(prompt)}")
            raise TypeError("prompt は str or ChatTemplate or list でなければなりません。")
            
        return self.model.invoke(messages)

    def bind_tools(self, tools: list):
        self.model = self.model.bind_tools(tools)

    def stockMessages(self, response: list):
        """
        responseを受け取り、メッセージを保持する
        """
        pass
    
## OpenAI を使ったモデルサプライヤ
class OpenAIManager(ModelBaseManager):
    def __init__(self, secrets, logger: logging.Logger, llmKind: str):
        self.logger = logger
        self.modelName = secrets[llmKind].get("modelName", "gpt-4o")
        self.client = OpenAI(
            base_url = secrets[llmKind]["baseUrl"],
            api_key = secrets[llmKind]["apiKey"],
        )
        self.sys_prompt = secrets[llmKind]["sys_prompt"]
        self.tools = None

    def bind_tools(self, tools: list):
        # LangChainのツールリストをOpenAIネイティブのJSONフォーマットに変換して保持
        from langchain_core.utils.function_calling import convert_to_openai_tool
        self.tools = [convert_to_openai_tool(tool) for tool in tools]

    def invoke(self, prompt: Union[str, ChatTemplate, list]):
        """
        ネイティブなOpenAI APIを使ってリクエストを行い、
        LangGraphが読めるようにLangChainのAIMessageフォーマットに変換して返す。
        """
        if isinstance(prompt, str):
            lc_messages = ChatTemplate(
                sys_prompt=self.sys_prompt,
                usr_prompt=prompt,
            ).mk_messages
        elif isinstance(prompt, ChatTemplate):
            lc_messages = prompt.mk_messages
        elif isinstance(prompt, list):
            lc_messages = prompt
        else:
            self.logger.error(f"不正なプロンプト形式です: {type(prompt)}")
            raise TypeError("prompt は str or ChatTemplate or list でなければなりません。")
            
        # 1. LangChainのメッセージリストを、OpenAI API用の辞書フォーマットに変換
        oai_messages = []
        for msg in lc_messages:
            if isinstance(msg, SystemMessage):
                oai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                oai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                d = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc["id"], 
                            "type": "function", 
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}
                        } for tc in msg.tool_calls
                    ]
                oai_messages.append(d)
            elif isinstance(msg, ToolMessage):
                oai_messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
            else:
                oai_messages.append({"role": "user", "content": str(msg.content)})

        # 2. OpenAI APIにリクエストを送信
        kwargs = {
            "model": self.modelName,
            "messages": oai_messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        
        # 3. OpenAIのネイティックレスポンスを、LangChainのAIMessage(tool_calls付き)に逆変換して返す
        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                    "id": tc.id
                })
                
        return AIMessage(content=choice.content or "", tool_calls=tool_calls)
    
    def stockMessages(self, response: list):
        pass
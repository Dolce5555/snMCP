import logging
from langchain.chat_models import init_chat_model
from openai import OpenAI
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, Union

####################

## プロンプトテンプレートを型規定
class ChatTemplate(BaseModel):
    sys_prompt: str
    usr_prompt: str
    ai_prompt: Optional[str] = None
    old_messages: Optional[list] = None

    @property
    def mk_messages(self):
        if self.old_messages:
            return self.old_messages + [
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=self.usr_prompt)
            ]
        else:
            return [
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=self.usr_prompt)
            ]

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
        self.logger = logger or self.getLogger(f"{__file__.split("/")[-1]}_{self.__class__.__name__}", logLevel)
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

####################

## 自分のつかうモデルサプライヤを抽象化
class ModelBaseManager(ABC):
    @abstractmethod
    def __init__(self, secrets, logger: logging.Logger, llmKind: str):
        pass
    @abstractmethod
    def invoke(self, prompt: Union[str, ChatTemplate]):
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
        try:
            self.invoke(prompt="接続テストです。")
            self.logger.info("LLMとの接続を確認できました。")
        except Exception as e:
            self.logger.error("LLMとの接続を確認できませんでした。%s", e)

    def invoke(self, prompt: Union[str, ChatTemplate]):
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
        else:
            self.logger.error(f"不正なプロンプト形式です: {type(prompt)}")
            raise TypeError("prompt は str or ChatTemplate でなければなりません。")
            
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
        """
        """
        self.logger = logger
        self.model = OpenAI(
            # model = secrets["baseLlm"]["modelName"],
            base_url = secrets["baseLlm"]["baseUrl"],
            api_key = secrets["baseLlm"]["apiKey"],
        )
        # self.embed_model = OpenAIEmbeddings(
        #     model = secrets["embeddingModel"]["modelName"],
        #     base_url = secrets["embeddingModel"]["baseUrl"],
        #     api_key = secrets["embeddingModel"]["apiKey"],
        # )
        # self.sys_prompt = secrets["baseLlm"]["sys_prompt"]
        # try:
        #     self.invoke(prompt="接続テストです。")
        #     self.logger.info("LLMとの接続を確認できました。")
        # except Exception as e:
        #     self.logger.error("LLMとの接続を確認できませんでした。%s", e)

    def invoke(self, prompt: Union[str, ChatTemplate]):
        # """
        # promptを受け取り、LLMに投げかける。
        # 文字列(usr_prompt)、ChatTemplateインスタンス、あるいは整形済みのメッセージリストを受け入れる。
        # """
        # if isinstance(prompt, str):
        #     messages = ChatTemplate(
        #         sys_prompt=self.sys_prompt,
        #         usr_prompt=prompt,
        #     ).mk_messages
        # elif isinstance(prompt, ChatTemplate):
        #     messages = prompt.mk_messages
        # else:
        #     self.logger.error(f"不正なプロンプト形式です: {type(prompt)}")
        #     raise TypeError("prompt は str or ChatTemplate でなければなりません。")
            
        # return self.gpu_model.invoke(messages)
        pass
    
    def stockMessages(self, response: list):
        """
        responseを受け取り、メッセージを保持する
        """
        pass
import logging
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

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

## 自分のつかうモデルサプライヤを抽象化
class ModelBaseManager(ABC):
    @abstractmethod
    def __init__(self, secrets, logger: logging.Logger = None, logLevel: int = logging.INFO):
        pass
    @abstractmethod
    def invoke(self, usr_prompt: str):
        pass
    @abstractmethod
    def stockMessages(self, response: list):
        pass
    @abstractmethod
    def getLogger(self, name: str = __name__, logLevel: int = logging.INFO) -> logging.Logger:
        pass

## VLLM を使ったモデルサプライヤ
class VLLMManager(ModelBaseManager):
    def __init__(self, secrets, logger: logging.Logger = None, logLevel: int = logging.INFO):
        """
        """
        self.logger = logger or self.getLogger(__name__, logLevel)
        self.gpu_model = init_chat_model(
            model = secrets["baseLlm"]["modelName"],
            base_url = secrets["baseLlm"]["baseUrl"],
            model_provider = "openai",
            api_key = secrets["baseLlm"]["apiKey"],
        )
        # self.embed_model = OpenAIEmbeddings(
        #     model = secrets["embeddingModel"]["modelName"],
        #     base_url = secrets["embeddingModel"]["baseUrl"],
        #     api_key = secrets["embeddingModel"]["apiKey"],
        # )
        self.sys_prompt = secrets["baseLlm"]["sys_prompt"]
        try:
            self.invoke(usr_prompt="接続テストです。")
            self.logger.info("LLMとの接続を確認できました。")
        except Exception as e:
            self.logger.error("LLMとの接続を確認できませんでした。%s", e)

    def invoke(self, usr_prompt: str):
        """
        usr_promptを受け取り、LLMに投げかける
        """
        return self.gpu_model.invoke(
            ChatTemplate(
                sys_prompt=self.sys_prompt,
                usr_prompt=usr_prompt,
            ).mk_messages
        )
    
    def stockMessages(self, response: list):
        """
        responseを受け取り、メッセージを保持する
        """
        pass
    
    def getLogger(self, name: str = __name__, logLevel: int = logging.INFO) -> logging.Logger:
        """
        シンプルなロガー初期化ヘルパー。
        アプリ側でハンドラを既に設定している場合は干渉しない。
        """        
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

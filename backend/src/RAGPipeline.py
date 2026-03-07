from WeaviateManager import WeaviateRAGSearcher, WeaviateCollectionManager, WeaviateDocumentManager
from ruamel.yaml import YAML
from LLMManager import LLMManager, ChatTemplate
from abc import ABC, abstractmethod
import logging

class BasePipeline(ABC):
    @abstractmethod
    def __init__(self, secrets, logger: logging.Logger):
        pass
    @abstractmethod
    def run(self, query: str):
        pass
    @abstractmethod
    def _query_format(self):
        pass

class RAGPipeline(BasePipeline):
    def __init__(self, secrets, logger: logging.Logger, usr_question: str, mode: str = "raw", history: list = None):
        self.logger = logger
        self.logger.info("RAGPipelineを初期化します。")
        self.ragSearcher = WeaviateRAGSearcher(secrets, "mariage_docs", self.logger)
        self.llm = LLMManager("vllm", secrets, self.logger)
        self.usr_question = usr_question
        self.mode = mode
        self.query = None
        self.search_result = None
        self.generated_response = None
        self.history = history

    def run(self):
        """
        メイン処理
        1. プロンプト整形
        2. ベクトルDB検索
        3. レスポンス生成
        4. ヒストリー更新
        """
        self._query_format()
        self.logger.info(f"検索クエリ: {self.query}")
        self.search_result = self.ragSearcher.contextSearch(self.query)
        self.logger.info(f"検索結果: {self.search_result}")
        self._generate_response()
        self.logger.info(f"生成されたレスポンス: {self.generated_response}")
        # self._update_history()

    def _query_format(self):
        if self.mode == "raw":
            self.query = self.usr_question
        elif self.mode == "simple":
            llm_query = ChatTemplate(
                sys_prompt="""\
                ユーザーの質問を下記の例を参考にベクトルDBの検索に適した形に書き換えなさい。回答はクエリのみを返すこと。\n\
                例：\n\
                ユーザーの質問：「おいしいお米の銘柄について教えてください。」\n\
                ベクトルDBの検索に適したクエリ：「お米 銘柄 おいしい」""",
                usr_prompt=self.usr_question,
            )
            self.query = self.llm.manager.invoke(llm_query).content
        else:
            raise ValueError("mode must be 'simple' or 'raw'")
    
    def _generate_response(self):
        prompt = ChatTemplate(
            sys_prompt=f"""\
            あなたはワタベウェディングのコンシェルジュです。ユーザーの質問に対して、検索結果を参考に回答してください。\n\
            検索結果:\n\
            {self.search_result}""",
            usr_prompt=self.usr_question,
            # old_messages=self.history,
        )
        self.generated_response = self.llm.manager.invoke(prompt).content


if __name__ == "__main__":
    # secrets.yamlを読み込む
    filePath = "/app/backend/secrets.yaml"
    yaml = YAML(typ = "safe", pure = True)
    with open(filePath, "r", encoding="utf-8") as f:
        secrets = yaml.load(f)
    # ロガーの初期化
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    # RAGPipelineの初期化
    ragPipeline = RAGPipeline(
        secrets,
        logger,
        """
        今、ワタベウェディングで結婚式を挙げようと考えています。
        もし、タキシードを持ち込もうと考えているのですが、持ち込みにかかる費用について教えてください。
        """,
        mode="simple"
    )
    ragPipeline.run()

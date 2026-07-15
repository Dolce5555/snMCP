import os
import time
import weaviate
import logging
from ruamel.yaml import YAML
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
from weaviate.classes.init import AdditionalConfig, Timeout, Auth
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.config import Configure, Property, DataType
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

##########
class WeaviateManageBase:
    """
    Weaviateに接続する基底クラス
    """
    def __init__(self, secrets, logger: logging.Logger = None, logLevel: int = logging.INFO):
        # secret情報を保持
        self.secrets = secrets
        
        # 外部からロガーが渡されなければモジュールロガーを作成
        try:
            self.logger = logger.getChild(self.__class__.__name__)
        except Exception as e:
            self.logger = self.getLogger(f"{__file__.split("/")[-1]}_{self.__class__.__name__}", logLevel)
        
        self.client = weaviate.connect_to_local(
            host = secrets["weaviate"]["host"],
            port = secrets["weaviate"]["httpPort"],
            grpc_port = secrets["weaviate"]["grpcPort"],
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)  # タイムアウトの設定（標準記載の転記のため詳細動作不明）
            ),
            auth_credentials=Auth.api_key(secrets["weaviate"]["apiKey"]["user-a"]),
            headers = {
                "X-OpenAI-Api-Key": secrets["embeddingModel"]["apiKey"],
            }
        )
        # check if Weaviate is ready
        while not self.client.is_ready():
            self.logger.debug("weaviateの準備ができるまで待機しています...")
            time.sleep(5)
        self.logger.info("Weaviateの準備ができました。")


    def __del__(self):
        if getattr(self, "client", None) and self.client.is_live():
            self.logger.info("weaviateとの接続を終了します。")
            self.client.close()


    def getLogger(self, name: str, logLevel: int = logging.INFO) -> logging.Logger:
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


##########
class WeaviateRAGSearcher(WeaviateManageBase):
    """
    Weaviateに接続し、オブジェクトを検索するクラス
    """
    def __init__(self, secrets, collectionName: str, logger: logging.Logger = None, logLevel: int = logging.INFO):
        super().__init__(secrets, logger, logLevel)
        try:
            self.collection = self.client.collections.use(collectionName)
        except Exception as e:
            self.logger.error(f"コレクションの接続に失敗しました: {e}")
            self.collection = None

    def contextSearch(self, query: str, method: str = "semantic"):
        result = ""
        if method == "semantic":
            self.logger.info("semantic searchを実行します。")
            response = self.collection.query.near_text(
                query = query,
                limit = 4,
                return_metadata = MetadataQuery(
                    distance = True,
                )
            )
            for i, obj in enumerate(response.objects):
                # result += f"{obj.properties.metadata}: [contents: {obj.properties.page_content}, distance: {obj.metadata.distance}]\n"
                result += f"No.{i + 1}: [properties: {obj.properties}, distance: {obj.metadata.distance}]\n"
            return result
        elif method == "keyword":
            self.logger.info("keyword searchを実行します。")
            return result
        elif method == "hybrid":
            return result
        else:
            self.logger.error(f"不明な検索方法が指定されました: {method}")
            return None


##########
class WeaviateCollectionManager(WeaviateManageBase):
    """
    Weaviateに接続し、 collection を管理するクラス
    """
    def __init__(self, secrets, logger: logging.Logger = None, logLevel: int = logging.INFO):
        super().__init__(secrets, logger, logLevel)


    def createCollection(self, collectionName: str):
        self.client.collections.create(
            name = collectionName,
            vector_config = Configure.Vectors.text2vec_openai(
                model = self.secrets["embeddingModel"]["modelName"],
                base_url = self.secrets["embeddingModel"]["baseUrl"],
                name = f"{collectionName}_vector",
            ),
        )
        self.logger.info(f"コレクションを作成しました。")
        self.logger.info(f"現在のコレクションは、下記のものがあります。\n{self.readCollection()}")
    

    def readCollection(self):
        collections = list(self.client.collections.list_all().keys())
        self.logger.info(f"現在のコレクションは、下記のものがあります。\n{collections}")


    def deleteCollection(self, collectionName: str = None, option: str = "one"):
        if option == "one":
            self.client.collections.delete(collectionName)
            self.logger.info(f"コレクションを削除しました。")
        elif option == "all":
            self.client.collections.delete_all()
            self.logger.info(f"コレクションを削除しました。")
        else:
            self.logger.error(f"不明なオプションが指定されました: {option}")
        self.logger.info(f"現在のコレクションは、下記のものがあります。\n{self.readCollection()}")


##########
class WeaviateDocumentManager(WeaviateManageBase):
    """
    Weaviateに接続し、 object を管理するクラス
    """
    def __init__(self, secrets, collectionName: str, logger: logging.Logger = None, logLevel: int = logging.INFO):
        super().__init__(secrets, logger, logLevel)
        self.collection = self.client.collections.use(collectionName)
        self.logger.info(f"コレクションの接続に成功しました: {collectionName}")


    def insertObject(self, filePath: str, chunkSize: int = 1000, chunkOverlap: int = 200):
        # doclingでpdfをmd形式に変換
        mdDoc = self.formatMd(filePath)
        fileName = filePath.split("\\")[-1]

        splitedMd = self.splitDoc(mdDoc, chunkSize, chunkOverlap)

        ## Weaviateに文書を挿入する
        with self.collection.batch.dynamic() as batch:
            for obj in splitedMd:
                if not obj.metadata:
                    batch.add_object(
                        properties = {
                            "file_name": fileName,
                            # "metadata": {"Header": "No data"},
                            "page_content": obj.page_content,
                        }
                    )
                else:
                    batch.add_object(
                        properties = {
                            "file_name": fileName,
                            "metadata": obj.metadata,
                            "page_content": obj.page_content,
                        }
                    )
        self.logger.info("データの挿入が完了しました。")


    def formatMd(self, filePath: str):
        # doclingでpdfをmd形式に変換
        ocrOptions = RapidOcrOptions(lang = ["ja", "en"])
        piplineOptions = PdfPipelineOptions()
        piplineOptions.ocr_options = ocrOptions
        piplineOptions.do_ocr = True  # OCR機能を有効化
        piplineOptions.do_table_structure = True  # 表構造認識を有効化
        piplineOptions.table_structure_options.do_cell_matching = True  # セルの認識方法: ビジュアルに基づいてセルを認識させる
        piplineOptions.table_structure_options.mode = TableFormerMode.ACCURATE  # 表の認識方法: 精密な表を認識する
        converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options = piplineOptions),
            }
        )
        # converter = DocumentConverter()
        mdDoc = converter.convert(filePath)
        return mdDoc


    def splitDoc(self, mdDoc, chunkSize: int, chunkOverlap: int):
        ## md形式の文書を分割する
        # md形式のヘッダ情報で分割ポイントを設定
        headers2SplitOn = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
        ]
        mdSplitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = headers2SplitOn,
            strip_headers = False,
        )
        # 分割するチャンク数と重複を設定
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = chunkSize,
            chunk_overlap = chunkOverlap
        )
        splitedMd = mdSplitter.split_text(mdDoc.document.export_to_markdown())
        splitedMd = textSplitter.split_documents(splitedMd)
        return splitedMd


    def readObjects(self, fileName: str = None):
        if fileName:
            objects = self.collection.query.fetch_objects(
                filters = Filter.by_property("file_name").equal(fileName)
            ).objects
            for obj in objects:
                self.logger.info(obj.uuid)
                self.logger.info(obj.properties)
        else:
            for obj in self.collection.iterator():
                self.logger.info(obj.uuid)
                self.logger.info(obj.properties)


if __name__ == "__main__":
    ## secret情報（api-keyや接続先など）をyamlから取得
    filePath = "/app/backend/secrets.yaml"
    yaml = YAML(typ = "safe", pure = True)
    with open(filePath, "r", encoding="utf-8") as f:
        secrets = yaml.load(f)
    
    wcm = WeaviateCollectionManager(secrets)
    wcm.deleteCollection("mariage_docs")
    wcm.createCollection("mariage_docs")
    wcm.readCollection()
    
    dirPath = "/app/backend/ragOriginalData/"
    fileName = "1593194_名倉様_持込品に関する注意事項.pdf"
    filePath = dirPath + fileName

    wdm = WeaviateDocumentManager(secrets, "mariage_docs")
    wdm.insertObject(filePath)
    wdm.readObjects(fileName)

    wrs = WeaviateRAGSearcher(secrets, "mariage_docs")
    result = wrs.contextSearch("タキシードの持ち込み費用はいくらかかりますか？")
    print(result)
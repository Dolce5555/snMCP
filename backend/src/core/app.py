from WeaviateManager import WeaviateRAGSearcher, WeaviateCollectionManager, WeaviateDocumentManager
from ruamel.yaml import YAML
from LLMManager import LLMManager
from RAGPipeline import RAGPipeline
import logging
print("app.pyが読み込まれました")
if __name__ == "__main__":
    ## secret情報（api-keyや接続先など）をyamlから取得
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
    
    # dirPath = "/app/backend/ragOriginalData/"
    # fileName = "1593194_名倉様_持込品に関する注意事項.pdf"
    # filePath = dirPath + fileName
    
    # wdm = WeaviateDocumentManager(secrets, "test")
    # # wdm.insertObject(filePath)
    # wdm.readObjects(fileName)

    # llm = LLMManager("vllm", secrets)
    # print(llm.manager.invoke("あなたは何を手伝ってくれますか？"))
    ragPipeline = RAGPipeline(
        secrets,
        logger,
        """\
        今、ワタベウェディングで結婚式を挙げようと考えています。\n\
        もし、タキシードを持ち込もうと考えているのですが、持ち込みにかかる費用について教えてください。""",
        mode="raw",
        pipeline_kind="agent"
        # pipeline_kind="2steps"
    )
    ragPipeline.run()
    # collections = ragPipeline.ragSearcher.client.collections.list_all() # 脱獄用
    # print(collections) # 脱獄用
    # print(f"\n\n実際にユーザに返すレスポンス内容\n{ragPipeline.generated_response}")
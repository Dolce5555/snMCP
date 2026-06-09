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
    fmt = "%(asctime)s %(levelname)s [%(filename)s : %(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    
    # dirPath = "/app/backend/ragOriginalData/"
    # fileName = "1593194_名倉様_持込品に関する注意事項.pdf"
    # filePath = dirPath + fileName
    
    # wdm = WeaviateDocumentManager(secrets, "test")
    # # wdm.insertObject(filePath)
    # wdm.readObjects(fileName)

    # 外部で依存オブジェクトを初期化
    llm = LLMManager("vllm", secrets, logger, llmKind="baseLlm")
    ragSearcher = WeaviateRAGSearcher(secrets, "mariage_docs", logger)

    # パイプラインを初期化（質問文はこの時点では渡さない）
    ragPipeline = RAGPipeline(
        llm=llm,
        ragSearcher=ragSearcher,
        logger=logger,
        pipeline_kind="agent"
    )
    
    # 実行
    question = "今、ワタベウェディングで結婚式を挙げようと考えています。\nもし、タキシードを持ち込もうと考えているのですが、持ち込みにかかる費用について教えてください。"
    response = ragPipeline.run(usr_question=question, mode="raw")
    print(f"\n最終出力結果:\n{response}")
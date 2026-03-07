from WeaviateManager import WeaviateRAGSearcher, WeaviateCollectionManager, WeaviateDocumentManager
from ruamel.yaml import YAML
from LLMManager import LLMManager

if __name__ == "__main__":
    ## secret情報（api-keyや接続先など）をyamlから取得
    filePath = "/app/backend/secrets.yaml"
    yaml = YAML(typ = "safe", pure = True)
    with open(filePath, "r", encoding="utf-8") as f:
        secrets = yaml.load(f)
    
    # dirPath = "/app/backend/ragOriginalData/"
    # fileName = "1593194_名倉様_持込品に関する注意事項.pdf"
    # filePath = dirPath + fileName
    
    # wdm = WeaviateDocumentManager(secrets, "test")
    # # wdm.insertObject(filePath)
    # wdm.readObjects(fileName)

    llm = LLMManager("vllm", secrets)
    print(llm.manager.invoke("あなたは何を手伝ってくれますか？"))
    
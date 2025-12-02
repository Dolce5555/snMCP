import time
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout, Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType
from ruamel.yaml import YAML

filePath = "/app/backend/secrets.yaml"
yaml = YAML(typ = "safe", pure = True)
with open(filePath, "r", encoding="utf-8") as f:
    secrets = yaml.load(f)

data_objects = [
    {"title": "The Matrix", "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.", "genre": "Science Fiction"},
    {"title": "Spirited Away", "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.", "genre": "Animation"},
    {"title": "The Lord of the Rings: The Fellowship of the Ring", "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.", "genre": "Fantasy"},
    {"title": "鉄腕アトム", "description": "未来の世界で、少年ロボットが人間と共に冒険を繰り広げる物語。", "genre": "アニメーション"},
    {"title": "千と千尋の神隠し", "description": "少女が不思議な世界で両親を救うために奮闘する冒険物語。", "genre": "アニメーション"},
    {"title": "君の名は。", "description": "都会に住む少年と田舎に住む少女が、夢の中で入れ替わる不思議な体験を通じて絆を深める物語。", "genre": "ロマンス"},
]
try:
    client = weaviate.connect_to_local(
        host = secrets["weaviate"]["host"],
        port = secrets["weaviate"]["httpPort"],
        grpc_port = secrets["weaviate"]["grpcPort"],
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
        ),
        auth_credentials=Auth.api_key(secrets["weaviate"]["apiKey"]["user-a"]),
        headers = {
            "X-OpenAI-Api-Key": secrets["embeddingModel"]["apiKey"],
        #     "X-OpenAI-Baseurl": f"http://{model_host}:9000/v1/embeddings",
            # "X-OpenAI-Baseurl": f"http://{model_host}:9000",
        }
    )
    
    # check if Weaviate is ready
    while not client.is_ready():
        print("Waiting for Weaviate to be ready...")
        time.sleep(5)
    print("Weaviate is ready.")

    # Create a collection with OpenAI( or VLLM) embeddings
    client.collections.delete_all()
    client.collections.create(
        name = "test",
        vector_config = Configure.Vectors.text2vec_openai(
            model = secrets["embeddingModel"]["modelName"],
            base_url = secrets["embeddingModel"]["baseUrl"],
            name = "test_vector",
            # type = "text",
        ),
    )

    # Insert data objects
    test_collection = client.collections.use("test")
    # with test_collection.batch.fixed_size(batch_size=200) as batch:
    with test_collection.batch.dynamic() as batch:
        for obj in data_objects:
            batch.add_object(properties = obj)
    print(len(test_collection))
    
    response = test_collection.query.near_text(
        query = "boy",
        limit = 4,
        return_metadata = MetadataQuery(distance=True, certainty=True, score=True, explain_score=True)
    )
    for obj in response.objects:
        print(obj)
finally:
    client.close()
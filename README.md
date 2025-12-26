# snMCP

## 初期セットアップ

### 前提条件
- 今回のコンテナは、WSL2環境で動作することは確認できています。
- AMDGPUを使う人は、vllmGPUのコンテナを有効にしてよいですが、あくまでWSL2でしか動作しない設定になっています。
- `VLLM`という、hugging-face のLLMをローカル環境で動かせるようにするソフトウェアを使う
    - 類似ソフトウェア：`ollama`, `huggingface/transformers`など
    - AMDのGPU環境は、vllmGPUのコンテナを起動する
    - CPU環境は、vllmCPUのコンテナを起動する
        - vllmGPUはコメントアウトしたままで！！

### セットアップ手順

1. __envs__/samples/ディレクトリ内のファイルをコピーして、__envs__直下に配置する
    - vllmXXX.env
        - HF_TOKEN: huggingfaceのトークンを取得して、記載する

1. docker-composeを起動する
    ```bash
    docker-compose up -d --build
    ```

1. vllmXXXコンテナに入る（XXXはGPUまたはCPU）
    ```bash
    docker exec -it vllmXXX /bin/bash
    ```

1. vllmXXXコンテナ内で、huggingfaceからモデルをロードする (*1)
    ```bash
    mkdir /app/vllm/models/<your-model-name: e.g. gemma-3-1b>
    huggingface-cli download <model-name: e.g. google/gemma-3-1b-it> --local-dir /app/vllm/models/<your-model-name: e.g. gemma-3-1b>
    ```
    - ロードが終われば、コンテナを抜ける
        ```bash
        exit
        ```

1. vllmXXXコンテナで、ロードしたモデルを起動できるようなconfigファイルを作成する
    - vllmCPUの場合
        - vllmCPU/models/example.yamlを参考に、同様の場所にconfigファイルを参照すればよい。 (e.g. gemma-3-1b.yaml)

1. vllmXXXコンテナの起動コマンドで、vllmの起動時にconfigを参照するようにdocker-composeで指定する
    ```yaml
    command: vllm serve --config /app/vllm/models/gemma-3-1b.yaml
    ```

1. vllmXXXコンテナを再起動する
    ```bash
    docker-compose up -d --build
    ```

1. vllmXXXコンテナのログで下記のような表示がされていれば、モデルが正しくデプロイされている（はず）
    ```bash
    docker logs vllmXXX
    ```
    ```bash
    ~~前略~~
    (APIServer pid=1) INFO:     Started server process [1]
    (APIServer pid=1) INFO:     Waiting for application startup.
    (APIServer pid=1) INFO:     Application startup complete.
    ```

### 動作確認

1. backendコンテナに入る
    ```bash
    docker exec -it backend /bin/bash
    ```

1. vllm_access.pyを修正する
    - ただし、下記はCPU版の例
    ```python
    # ~~前略~~
    self.cpu_model = init_chat_model(
            model = "gemma-3-1b",
            base_url = "http://vllmCPU:9000/v1",
            model_provider = "openai",
            api_key = "EMPTY",
        )
    # ~~後略~~
    ```

1. vllm_access.pyを実行し、下記のような出力がされること
    ```bash
    python vllm_access.py
    ```
    ```bash
    content='こんにちは！元気ですよ！😊 あなたは元気ですか？何かお手伝いできることはありますか？' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 24, 'total_tokens': 45, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'gemma-3-1b', 'system_fingerprint': None, 'id': 'chatcmpl-98e0e7d10286523b', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--b5e87f13-9aa3-4060-b29a-8082fcee760d-0' usage_metadata={'input_tokens': 24, 'output_tokens': 21, 'total_tokens': 45, 'input_token_details': {}, 'output_token_details': {}}
    ```

## 2回目以降の起動

1. docker-composeを起動する
    ```bash
    docker-compose up -d --build
    ```

1. vllmXXXのログが正しく出力されていること
    ```bash
    docker logs vllmXXX
    ```
    ```bash
    (APIServer pid=1) INFO:     Started server process [1]
    (APIServer pid=1) INFO:     Waiting for application startup.
    (APIServer pid=1) INFO:     Application startup complete.
    ```
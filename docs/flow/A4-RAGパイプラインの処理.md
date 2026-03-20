# RAGPipeline API リファレンス

## 1. 概要 (Overview)
[RAGPipeline](../../backend/src/RAGPipeline.py:17:0-71:73) は、ユーザーからの質問を受け取り、Weaviate（ベクトルデータベース）での類似文書検索と、vLLM（大規模言語モデル）による回答生成を一貫して行うためのパイプラインクラスです。
[BasePipeline](../../backend/src/RAGPipeline.py:6:0-15:12) 抽象クラスを継承して実装されています。

---

## 2. 初期化 (Constructor)

### [RAGPipeline(secrets, logger, usr_question, mode="raw", history=None)](../../backend/src/RAGPipeline.py:17:0-71:73)

RAGパイプラインのインスタンスを生成し、LLMManager および WeaviateRAGSearcher の初期化を行います。

#### 引数 (Parameters)
| 引数名 | 型 | デフォルト値 | 説明 |
| :--- | :---: | :---: | :--- |
| `secrets` | `dict` | (必須) | DBやLLMの接続情報が記載された辞書（[secrets.yaml](../../backend/secrets.yaml:0:0-0:0)の内容） |
| `logger` | `logging.Logger` | (必須) | アプリケーションのロガーインスタンス |
| `usr_question` | `str` | (必須) | ユーザーから入力された生の質問文 |
| `mode` | `str` | `"raw"` | 検索クエリの処理モード。（後述の「モードについて」を参照）|
| `history` | `list` | `None` | (未実装) 過去の会話履歴を保持するリスト |

#### モードについて (`mode`)
- `"raw"`: `usr_question` をそのままベクトルDBの検索クエリとして使用します。
- `"simple"`: LLMを使用して `usr_question` を「ベクトルDBの検索に適した単語の羅列（キーワード）」に自動で書き換えてから検索を行います。

---

## 3. メソッド (Methods)

### [run()](../../backend/src/RAGPipeline.py:10:4-12:12)
パイプライン処理のメインとなる実行メソッドです。呼び出すことで、プロンプトの整形、文書の検索、LLMによる回答生成までのすべてのステップを順次実行します。

- **戻り値**: なし（結果はインスタンスの属性に保存されます）
- **内部処理の流れ**:
  1. [_query_format()](../../backend/src/RAGPipeline.py:13:4-15:12): 質問文を検索クエリに変換。
  2. `ragSearcher.contextSearch()`: 生成したクエリでWeaviateから関連情報を検索。
  3. [_generate_response()](../../backend/src/RAGPipeline.py:62:4-71:73): 検索結果を基に、ワタベウェディングのコンシェルジュとして回答を生成。

---

## 4. プロパティ (Attributes)
実行後（[run()](../../backend/src/RAGPipeline.py:10:4-12:12) メソッドの呼び出し後）、以下の属性から処理の途中経過や最終結果を取得できます。

| 属性名 | 型 | 説明 |
| :--- | :---: | :--- |
| `query` | `str` | 実際にベクトルDBの検索に使用された整形後のクエリ文字列 |
| `search_result` | `str` | Weaviateから取得した関連文書（チャンク）の検索結果文字列 |
| `generated_response` | `str` | LLMが生成した最終的な回答テキスト |

---

## 5. 使用例 (Usage Example)

```python
import logging
from ruamel.yaml import YAML
from src.RAGPipeline import RAGPipeline

# 1. 準備 (secretsの読み込みとロガーの設定)
with open("secrets.yaml", "r", encoding="utf-8") as f:
    yaml = YAML(typ="safe", pure=True)
    secrets = yaml.load(f)
logger = logging.getLogger("AppLogger")

# 2. ユーザーの質問
question = "タキシードを持ち込もうと考えているのですが、持ち込みにかかる費用について教えてください。"

# 3. パイプラインの初期化と実行
pipeline = RAGPipeline(
    secrets=secrets,
    logger=logger,
    usr_question=question,
    mode="simple"  # 質問を検索キーワードに自動変換するモード
)
pipeline.run()

# 4. 結果の取得
print("--- 検索クエリ ---")
print(pipeline.query)

print("--- 最終的な回答 ---")
print(pipeline.generated_response)
```

import os
import time
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout, Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


documents = []
pdf_count = 0

for file_name in os.listdir(pdf_dir):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, file_name)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if docs:
                documents.extend(docs)
                pdf_count += 1
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

print(f"PDFファイル数: {pdf_count}")
print(f"総ドキュメントページ数: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)
texts = text_splitter.split_documents(documents)
print(f"分割後のテキストチャンク数: {len(texts)}")
print(f"テキストチャンクの例:{texts[:2]}")

embedding_model = ""
vector_db_path = ""
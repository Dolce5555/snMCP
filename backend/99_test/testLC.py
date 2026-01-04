from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

dirPath = "/app/backend/ragOriginalData"
filePath = dirPath + "/1593194_名倉様_お内金ご請求書.pdf"

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False  # OCR機能を有効化
pipeline_options.do_table_structure = True  # 表構造認識を有効化
converter = DocumentConverter(
    format_options={
        "pdf": pipeline_options,
    }
)
converter = DocumentConverter()
result = converter.convert(filePath)

headers2SplitOn = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

mdSplitter = MarkdownHeaderTextSplitter(
    headers_to_split_on = headers2SplitOn,
    strip_headers = False,
)
textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=100
)
splitedMd = mdSplitter.split_text(result.document.export_to_markdown())
print(splitedMd)
print("\n-----\n")
splitedMd = textSplitter.split_documents(splitedMd)
print(splitedMd)
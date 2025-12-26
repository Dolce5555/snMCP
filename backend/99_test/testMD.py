from markitdown import MarkItDown
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

dirPath = "/app/backend/ragOriginalData"
filePath = dirPath + "/1593194_名倉様_お内金ご請求書.pdf"

## how to MarkItDown
# mid = MarkItDown()
# result = mid.convert(filePath)
# print(result.text_content)

## how to docling
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # OCR機能を有効化
pipeline_options.do_table_structure = True  # 表構造認識を有効化
converter = DocumentConverter(
    format_options={
        "pdf": pipeline_options,
    }
)
converter = DocumentConverter()
result = converter.convert(filePath)
print(result.document.export_to_markdown())
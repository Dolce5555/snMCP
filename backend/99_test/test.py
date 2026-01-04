objects = [
    {
        "metadata": {"source": "doc1"},
        "page_content": "This is the content of document 1."
    },
    {
        "metadata": {"source": "doc2"},
        "page_content": "This is the content of document 2."
    },
    {
        "metadata": None,
        "page_content": "This is the content of document 3."
    }
]

for obj in objects:
    ojb_ = getattr(obj, "metadata", None) or {}
    print(ojb_)
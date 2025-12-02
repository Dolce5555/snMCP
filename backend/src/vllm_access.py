from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.vectorstores import InMemoryVectorStore

class VLLMManager:
    def __init__(self, sys_prompt: str = "あなたは有能なアシスタントです。"):
        self.model = init_chat_model(
            model = "google/gemma-3-1b-it",
            base_url = "http://vllmGPU:9000/v1",
            model_provider = "openai",
            api_key = "EMPTY",
        )
        self.embed_model = OpenAIEmbeddings(
            model = "embeddinggemma-300m",
            base_url = "http://vllmCPU:9000/v1",
            api_key = "EMPTY",
        )
        self.sys_prompt = sys_prompt

    def prompt_template(self, usr_prompt: str):
        messages = [
            SystemMessage(content=self.sys_prompt),
            HumanMessage(content=usr_prompt)
        ]
        return messages

if __name__ == "__main__":
    vllm_manager = VLLMManager()
    usr_prompt = "こんにちは、元気ですか？"
    text = "LangChain is the framework for building context-aware reasoning applications"

    # messages = vllm_manager.prompt_template(usr_prompt)
    # response = vllm_manager.model.generate(messages)
    response = vllm_manager.embed_model.embed_query("hello")
    print(response[:5])
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage, AIMessage

class VLLMManager:
    def __init__(self, sys_prompt: str = "あなたは有能なアシスタントです。"):
        self.gpu_model = init_chat_model(
            model = "gemma-3-1b",
            base_url = "http://vllmGPU:9000/v1",
            model_provider = "openai",
            api_key = "key-generator",
        )
        # self.cpu_model = init_chat_model(
        #     model = "<your model name>",
        #     base_url = "http://vllmCPU:<your model port>/v1",
        #     model_provider = "openai",
        #     api_key = "EMPTY",
        # )
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
    
    def invoke(self, usr_prompt: str, model_server: str = "gpu"):
        if model_server == "cpu":
            model = self.cpu_model
        elif model_server == "gpu":
            model = self.gpu_model
        elif model_server == "embed":
            model = self.embed_model
        else:
            raise ValueError("model_server must be 'cpu' or 'gpu' or 'embed'")
        response = model.invoke(self.prompt_template(usr_prompt))
        return response

if __name__ == "__main__":
    vllm_manager = VLLMManager()
    usr_prompt = "こんにちは、元気ですか？"
    response = vllm_manager.invoke(usr_prompt=usr_prompt)
    print(response)
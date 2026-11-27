from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "do_sample": False,
        "return_full_text": False,
    },
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of Nepal?")
print(result.content)

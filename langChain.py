#langChain
## https://teddylee777.github.io/langchain/langchain-tutorial-02/
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline

import os

with open(".hkey","r") as key_file:
        key = key_file.read()


os.environ['HUGGINGFACEHUB_API_TOKEN'] = key
os.environ['HF_HOME'] = '/home/i4624/Desktop/langChain'

repo_id = 'beomi/llama-2-ko-7b'

# 질의내용
question = "who is Son Heung Min?"

# 템플릿
template = """Question: {question}

Answer: """

# 프롬프트 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=["question"])

# HuggingFaceHub 객체 생성
llm =  HuggingFacePipeline.from_model_id(
    model_id=repo_id, 
    device=-1,
    task="text-generation",
    model_kwargs={"temperature": 0.2, 
                  "max_length": 512}
)

# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
print(llm_chain.run(question=question))


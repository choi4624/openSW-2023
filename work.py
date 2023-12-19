import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey



## openai model 

from openai import OpenAI

client = OpenAI()

### full original data printing 
## print(client.models.list())

model_list = []

for m in client.models.list():
    model_list.append(m.id)

model_list = sorted(model_list)

for m in model_list:
    print(m)
    

### test question 

from langchain.chat_models import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(temperature=0,               # 창의성 (0.0 ~ 2.0) 
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-3.5-turbo',  # 모델명
                )

# 질의내용
question = '미국의 수도는 뉴욕이야?'

# 질의
print(f'[답변]: {llm.predict(question)}')


### huggingface model 

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
# os.environ['HF_HOME'] = '/home/i4624/Desktop/langChain'

repo_id = 'mistralai/Mistral-7B-v0.1'


# 질의내용
question = "does python running as notebook?"

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
    model_kwargs={"max_length": 512}
)

# LLM Chain 객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실행
print(llm_chain.run(question=question))

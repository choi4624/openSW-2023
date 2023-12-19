import os

with open(".openaikey", "r") as key_file:
    openaikey = key_file.read()

# os.environ["OPENAI_API_KEY"] = openaikey

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# 설치: pip install python-dotenv
from dotenv import load_dotenv

# 토큰 정보로드
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

with open(".hkey", "r") as key_file:
    key = key_file.read()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = key
    os.environ["HF_HOME"] = "/home/i4624/Desktop/langChain"

import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

# 아티클 주소
url = "https://choi4624.github.io/computer/2023/07/12/NAB%EC%9D%B4%EC%83%81%ED%83%90%EC%A7%80.html"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, 'lxml')

# 뉴스 본문 내용
# news_content = soup.select_one("#newsct_article").text

print('web')


# 뉴스기사의 본문을 Chunk 단위로 쪼갬
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1024,  # 쪼개는 글자수
    chunk_overlap=300,  # 오버랩 글자수
    length_function=len,
    is_separator_regex=False,
)

# 웹사이트 내용 크롤링 후 Chunk 단위로 분할
docs = WebBaseLoader(url).load_and_split(text_splitter)
docs

# 각 Chunk 단위의 템플릿
template = '''다음의 내용을 한글로 요약해줘:

{text}
'''

# 전체 문서(혹은 전체 Chunk)에 대한 지시(instruct) 정의
combine_template = '''{text}

요약의 결과는 다음의 형식으로 작성해줘:
제목: 해당 아티클의 제목
주요내용: 한 줄로 요약된 내용
내용: 주요내용을 불렛포인트 형식으로 작성
'''



# 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=['text'])
combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'])


'''
이 아래부터 코드 수정영역
'''
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

print('model')
# llama-2-ko-7b model 이용 
repo_id = "beomi/llama-2-ko-7b"
model_id = 'beomi/llama-2-ko-7b'
checkpoint_name = "main"
checkpoint_path = "models/llama-2-ko-7b/"

## HF pipeline을 통해 로컬에서 실행하도록 설정 
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id, 
    task="text-generation", 
    model_kwargs={"temperature": 0.5, 
                  "max_length": 512}
)

print('summerize')

# 요약을 도와주는 load_summarize_chain
chain = load_summarize_chain(llm, map_prompt=prompt, combine_prompt=combine_prompt, chain_type="map_reduce", verbose=True)

print('result')
# print(docs)
# 실행결과
print(chain.run(docs))
print('end')
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

## 본문만 크롤링하게 변경

import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

# 네이버 뉴스기사 주소
url = "https://n.news.naver.com/mnews/article/031/0000797308?sid=105"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, 'lxml')

# 뉴스 본문 내용
news_content = soup.select_one("#newsct_article").text

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

print(docs)
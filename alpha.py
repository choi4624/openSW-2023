from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import os 

# PDF 파일 로드
loader = PyPDFLoader("./langchain/noukome-fix3.pdf")
document = loader.load()

document[0].page_content[:500]

# ========== ② 문서분할 ========== #

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n \n',  # 분할기준
    chunk_size=256,   # 사이즈
    chunk_overlap=0, # 중첩 사이즈
)

# 분할 실행
split_docs = text_splitter.split_documents(document)
# 총 분할된 도큐먼트 수
print(f'총 분할된 도큐먼트 수: {len(split_docs)}')

print(split_docs)
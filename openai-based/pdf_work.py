import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

# 토큰 정보로드
load_dotenv()
from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드
loader = PyPDFLoader("./data/reader.pdf")
document = loader.load()
document[0].page_content[:200] # 내용 추출

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(document)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 임베딩
embeddings = OpenAIEmbeddings()
# Chroma DB 에 저장
docsearch = Chroma.from_documents(texts, embeddings)
# retriever 가져옴
retriever = docsearch.as_retriever()

# langchain hub 에서 Prompt 다운로드 예시
# https://smith.langchain.com/hub/rlm/rag-prompt

from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt


# LLM
from langchain.chat_models import ChatOpenAI

# ChatGPT 모델 지정
llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)


# RAG chain 생성
from langchain.schema.runnable import RunnablePassthrough

# pipe operator를 활용한 체인 생성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rag_chain.invoke("please name of this novel and who is the main character?")

rag_chain.invoke("please short the this novel, may be 2-3paragraph")
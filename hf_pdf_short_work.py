from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey
# ========== ① 문서로드 ========== #

# PDF 파일 로드
loader = PyPDFLoader("./langchain/noukome.pdf")
document = loader.load()
document[0].page_content[:200]

# ========== ② 문서분할 ========== #

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_text(page_separator, chunk_size=3000, chunk_overlap=500)

# 분할 실행
split_docs = text_splitter.split_documents(document)
# 총 분할된 도큐먼트 수
print(f'총 분할된 도큐먼트 수: {len(split_docs)}')

# ========== ③ Map 단계 ========== #

# Map 단계에서 처리할 프롬프트 정의
# 분할된 문서에 적용할 프롬프트 내용을 기입합니다.
# 여기서 {pages} 변수에는 분할된 문서가 차례대로 대입되니다.
map_template = """다음은 문서 중 일부 내용입니다
{pages}
이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
답변:"""



# First model (map generation)
llm_map = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')
# Map 프롬프트 완성
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm_map, prompt=map_prompt)

print(map_template)

# ========== ④ Reduce 단계 ========== #

# Reduce 단계에서 처리할 프롬프트 정의
reduce_template = """다음은 요약의 집합입니다:
{doc_summaries}
이것들을 바탕으로 통합된 요약을 만들어 주세요.
답변:"""

print(reduce_template)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# llama-2-ko-7b model 이용 
repo_id = "beomi/llama-2-ko-7b"
model_id = 'beomi/llama-2-ko-7b'
checkpoint_name = "main"
checkpoint_path = "models/llama-2-ko-7b/"

## HF pipeline을 통해 로컬에서 실행하도록 설정 
llm_custom = HuggingFacePipeline.from_model_id(
    model_id=model_id, 
    task="text-generation", 
    model_kwargs={"temperature": 0.5, 
                  "max_length": 512}
)

# llm = ChatOpenAI(temperature=0, 
#                  model_name='gpt-3.5-turbo-16k')
# Reduce 프롬프트 완성
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce에서 수행할 LLMChain 정의
reduce_chain = LLMChain(llm=llm_custom, prompt=reduce_prompt)

# 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,                
    document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
)

# Map 문서를 통합하고 순차적으로 Reduce합니다.
reduce_documents_chain = ReduceDocumentsChain(
    # 호출되는 최종 체인입니다.
    combine_documents_chain=combine_documents_chain,
    # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
    collapse_documents_chain=combine_documents_chain,
    # 문서를 그룹화할 때의 토큰 최대 개수입니다.
    token_max=4000,
)

# ========== ⑤ Map-Reduce 통합단계 ========== #

# 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
map_reduce_chain = MapReduceDocumentsChain(
    # Map 체인
    llm_chain=map_chain,
    # Reduce 체인
    reduce_documents_chain=reduce_documents_chain,
    # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
    document_variable_name="pages",
    # 출력에서 매핑 단계의 결과를 반환합니다.
    return_intermediate_steps=False,
)

# ========== ⑥ 실행 결과 ========== #

# Map-Reduce 체인 실행
# 입력: 분할된 도큐먼트(②의 결과물)
result = map_reduce_chain.run(split_docs)
# 요약결과 출력
print(result)
''' 
###### base code ###########
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_K_M.gguf")
output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)

'''

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey
# ========== ① 문서로드 ========== #

# PDF 파일 로드
loader = PyPDFLoader("./langchain/short.pdf")
document = loader.load()

document[0].page_content[:500]

# ========== ② 문서분할 ========== #

# 스플리터 지정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator='\n \n',  # 분할기준
    chunk_size=256,   # 사이즈
    chunk_overlap=64, # 중첩 사이즈
)

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


## 맵 영역에서 라마 cpp 사용 시도 

# llm = Llama(model_path="./models/llama-2-7b-chat.Q4_K_M.gguf")
# First model (map generation)
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm_map = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    input={"temperature": 0.15, "max_length": 1024, "top_p": 1},
    callback_manager=callback_manager,
    verbose=False,
)
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

# # llama-2-ko-7b model 이용 
# repo_id = "beomi/llama-2-ko-7b"
# model_id = 'beomi/llama-2-ko-7b'
# checkpoint_name = "main"
# checkpoint_path = "models/llama-2-ko-7b/"

# ## HF pipeline을 통해 로컬에서 실행하도록 설정 
# llm_custom = HuggingFacePipeline.from_model_id(
#     model_id=model_id, 
#     task="text-generation", 
#     model_kwargs={"temperature": 0.5, 
#                   "max_length": 512}
# )

llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    input={"temperature": 0.15, "max_length": 1024, "top_p": 1},
    callback_manager=callback_manager,
    verbose=False,
)
# Reduce 프롬프트 완성
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Reduce에서 수행할 LLMChain 정의
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

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
    token_max=384,
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
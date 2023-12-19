import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# LLM 생성
chat = ChatOpenAI()

# 실행
chat(
    [HumanMessage(content="다음을 영어로 번역해 줘: 나는 프로그래밍을 정말 좋아해요")]
)

messages = [
    # SystemMessage: 역할 부여
    SystemMessage(content="너는 한글을 영어로 번역해 주는 전문번역가야."),
    # 질의
    HumanMessage(content="나는 프로그래밍을 정말 좋아해요")
]

# 실행
chat(messages)

### Converstation 

from langchain.chains import ConversationChain  

# ConversationChain 객체 생성
conversation = ConversationChain(llm=chat)

# 1차 질의
print(conversation.run("다음을 영어로 번역해 줘: 나는 프로그래밍을 정말 좋아해요"))

# 2차 질의
print(conversation.run("나는 운동도 정말 좋아해요"))

from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 1) 시스템 설정: 역할부여 정의
system_template = SystemMessagePromptTemplate.from_template(
    "너는 회사의 마케팅 담당자야. 다음의 상품 정보를 바탕으로 고객 응대를 해줘.\n{product}"
)
system_message = system_template.format(product="""
상품명: 하늘을 나는 스마트폰
제품가격: 300만원
주요기능:
- 스마트폰을 공중에 띄울 수 있음
- 방수기능
- 최신식 인공지능 솔루션 탑재
- 전화통화
- 음악
- 인터넷 브라우징, 5G 네트워크, 와이파이, 블루투스
제조사: 파인애플
""")

# 2) ChatPromptTemplate 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    system_message,                                              # 역할부여
    MessagesPlaceholder(variable_name="chat_history"),           # 메모리 저장소 설정. ConversationBufferMemory의 memory_key 와 동일하게 설정
    HumanMessagePromptTemplate.from_template("{human_input}"),   # 사용자 메시지 injection
])


# 3) LLM 모델 정의
llm = ChatOpenAI(model_name='gpt-3.5-turbo', 
                 temperature=0.5,
                 streaming=True,
                 callbacks=[StreamingStdOutCallbackHandler()])

# 4) 메모리 정의
memory = ConversationSummaryBufferMemory(llm=llm, 
                                         memory_key="chat_history", 
                                         max_token_limit=10, 
                                         return_messages=True
                                        )



# 5) LLMChain 정의
conversation = LLMChain(
    llm=llm,       # LLM
    prompt=prompt, # Prompt
    verbose=True,  # True 로 설정시 로그 출력
    memory=memory  # 메모리
)

# 6) 실행
answer = conversation({"human_input": "저에게 추천해 줄만한 상품이 있나요?"})
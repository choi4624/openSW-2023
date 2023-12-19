import os 

with open(".openaikey","r") as key_file:
        openaikey = key_file.read()
        
os.environ['OPENAI_API_KEY'] = openaikey

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# LLM 생성
chat = ChatOpenAI()

import pandas as pd

# csv 파일을 데이터프레임으로 로드
df = pd.read_csv('./data/titanic_dataset.csv')
df.head()

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-4-0613'),        # 모델 정의
    df,                                    # 데이터프레임
    verbose=True,                          # 추론과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# 질의
agent.run('데이터의 행과 열의 갯수는 어떻게 돼?')

# 질의
agent.run('남자 승객의 생존율을 어떻게 돼? %로 알려줘')

# 질의
agent.run('나이가 15세 이하인 승객중 1,2등급에 탑승한 남자 승객의 생존율은 어떻게 돼? %로 알려줘')

# 샘플 데이터프레임 생성
df1 = df.copy()
df1 = df1.fillna(0)
df1.head()


# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, 
               model='gpt-4-0613'),
               [df, df1], 
               verbose=True
)

# 질의
agent.run('나이 컬럼의 나이의 평균차이는 어떻게 돼? %로 구해줘.')
## langchain 구축 프로젝트 설명 

일단 로컬 환경에서 생각보다 잘 돌아가지 않음 
많이 어려움... 
llama CPP 를 거의 반드시 써야 함 
그리고 llama CPP 내의 제약조건으로 인해, 생각보다 큰 데이터를 그냥 집어넣기 어려운 부분도 있음 

alpha.py -> pdf tokenizer 테스트 
hf3_crawl_work.py -> beomi/llama-2-ko-7b 기반의 huggingFace model 로 수행한 웹 사이트 통째로 가져온 다음, 모델을 통해 요약 

hf_pdf_short_work.py -> llamaCPP 적용 이전의 huggingFace 사용 PDF 요약 시도 MAP 단계에서는 openAI를 이용하고 reduce는 hf 모델을 이용하도록 설정해본 시도 


cpp_hf_pdf_short_work.py -> llamaCPP 적용하여 map-reduce 방식으로 PDF 요약 기능 

langChain.py -> 최종적으로 4페이지 정도의 PDF 내용에 대한 질의 및 응답을 할 수 있도록 만들어진 코드 
* 다른 python 코드와 다르게 llama-index 로컬 버전으로 이용 
* embedding 모델은 BAAI/bge-base-en-v1.5 
https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html 

work.py -> 다른 작업 이전에, huggingFace 모델을 불러와서 로컬에서 실행해본 시도 "HuggingFacePipeline" 을 통해 로컬에서 실행 

사용한 모델 
llama-2-7b-chat.Q4_K_M.gguf 
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF 
성능적인 문제가 굉장히 강해서, 부득이 약간 수준의 모델을 이용 (대략 6GiB의 메모리 사용)



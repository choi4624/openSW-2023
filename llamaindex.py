from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from typing import Optional, List, Mapping, Any

from llama_index import ServiceContext, SimpleDirectoryReader, SummaryIndex
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from llama_index import GPTVectorStoreIndex
from llama_index import SimpleDirectoryReader

with open(".openaikey","r") as keyfile:
    key = keyfile.read()
import os
os.environ["OPENAI_API_KEY"] = key

import logging 
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# 문서 로드(data 폴더에 문서를 넣어 두세요)
documents = SimpleDirectoryReader("langchain").load_data()
from openai import OpenAI

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
from llama_index import GPTVectorStoreIndex

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-base-en-v1.5"
)

# Load the your data
documents = SimpleDirectoryReader("langchain").load_data()
print(documents)
index = SummaryIndex.from_documents(documents, service_context=service_context)

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query("please short the data in one sentence")
print(response)
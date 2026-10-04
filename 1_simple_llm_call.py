from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# llm = ChatOpenAI(
#     model="gemini-2.5-flash-lite",
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=GEMINI_API_KEY
# )
#
# result = llm.invoke("What is the capital of India?")
# print(result.content)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     google_api_key=GEMINI_API_KEY,
#     # max_output_tokens = 100
# )
#
# result = llm.invoke("What is the capital of India?")
# print(result.content)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

HF_TOKEN = os.getenv("HF_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1000,
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India? Explain.")
print(result.content)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)

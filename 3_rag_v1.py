# pip install -U langchain langchain-community langchain-huggingface \
#     sentence-transformers faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

PDF_PATH = "islr.pdf"


# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)


# 3. Create open-source embeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create FAISS vector database
vs = FAISS.from_documents(
    splits,
    emb
)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# 5. Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not found in the context, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


# 6. Hugging Face LLM
HF_TOKEN = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1000,
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm)


# 7. Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# 8. RAG pipeline
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = (
    parallel
    | prompt
    | model
    | StrOutputParser()
)


# 9. Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

while True:
    q = input("\nQ: ")

    if q.lower() in ["exit", "quit"]:
        break

    ans = chain.invoke(q.strip())

    print("\nA:", ans)
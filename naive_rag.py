"""1. Naive RAG LangChainnel: PDF -> oldalak -> fix méretű darabok -> vektortár -> top-k -> LLM.

    python naive_rag.py
    python naive_rag.py "Mit ellenőriz az érvényesítő?"

A beállítások a .env-ből jönnek (Azure OpenAI: gpt-5.4-mini + text-embedding-3-small).
Figyeld meg a forrásokat: melyik kérdésre jó a válasz, melyikre nem, és miért?
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF = HERE / "bme_kotelezettsegvallalas_rendje_2026.pdf"
load_dotenv(HERE / ".env")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Válaszolj a kérdésre kizárólag az alábbi kontextus alapján, magyarul, tömören. "
               "Ha a válasz nincs benne a kontextusban, mondd azt, hogy a dokumentum ezt nem tartalmazza.\n\n"
               "Kontextus:\n{context}"),
    ("human", "{question}"),
])

SAMPLE_QUESTIONS = [
    "Ki jegyzi ellen pénzügyileg egy nettó 8 millió forintos beszerzési szerződést?",
    "Ki vállal kötelezettséget egy kar 20 millió forintos marketingkampányánál?",
    "Ki dönt egy 100 millió forintot meghaladó peren kívüli egyezségről?",
]


def get_llm():
    # gpt-5.x: reasoning modell – temperature-t, max_tokens-t nem adunk meg (a LangChain el is hagyja)
    return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                           model=os.environ.get("AZURE_OPENAI_CHAT_MODEL", "gpt-5.4-mini"),
                           api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))


def get_embeddings():
    return AzureOpenAIEmbeddings(azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                                 model=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                                 api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))


def load_pages() -> list[Document]:
    """Oldalanként egy Document – a fejléc, lábléc, táblázat úgy, ahogy a pypdf kinyeri."""
    reader = PdfReader(PDF)
    return [Document(page_content=p.extract_text() or "", metadata={"page": i + 1})
            for i, p in enumerate(reader.pages)]


def build(top_k: int = 4):
    """Felépíti az indexet, és visszaad egy ask(kérdés) -> {answer, contexts, sources} függvényt."""
    # 1) darabolás: fix méret, átfedéssel – nem tudja, hol kezdődik egy § vagy egy táblázatsor
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(load_pages())
    # 2) embedding + vektortár (memóriában)
    store = InMemoryVectorStore.from_documents(chunks, embedding=get_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    # 3) generálás: prompt | LLM | szöveg
    chain = PROMPT | get_llm() | StrOutputParser()

    def ask(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(f"[{d.metadata['page']}. oldal]\n{d.page_content}" for d in docs)
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "contexts": [d.page_content for d in docs],
                "sources": [f"{d.metadata['page']}. old." for d in docs]}

    return ask


def main(build_fn, title: str):
    print(f"== {title} – index építése...", flush=True)
    ask = build_fn()
    for q in sys.argv[1:] or SAMPLE_QUESTIONS:
        r = ask(q)
        print(f"\n❓ {q}\n💬 {r['answer'].strip()}\n📎 {', '.join(r['sources'])}")


if __name__ == "__main__":
    main(build, "Naive RAG")

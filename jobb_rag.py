"""2. Jobb RAG LangChainnel: struktúra szerinti darabolás + kontextus-dúsítás + hibrid keresés (BM25 + vektor).

    python jobb_rag.py
    python jobb_rag.py "Ki dönt egy 100 millió forintot meghaladó peren kívüli egyezségről?"

Mi változott a naive_rag.py-hoz képest (lásd N1_3, 30–31. dia):
  1. tisztítás: az ismétlődő fejléc/lábléc/tartalomjegyzék kimarad,
  2. darabolás jelentés szerint: egy § = egy egység, a mellékletben egy táblázatsor = egy egység (oszlopnevekkel),
  3. kontextus-dúsítás: minden darab elejére odakerül, melyik §-ból / sorból jön, és mi a címe,
  4. hibrid keresés: BM25 (kulcsszó, magyar szótövezéssel) + vektoros keresés, rangsor-egyesítés (RRF),
  5. prompt: magyar, forrásmegjelölést kér, és megmondja, ha nincs válasz.
"""
import re

import Stemmer
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from naive_rag import PDF, get_embeddings, get_llm, main

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Egy egyetemi szabályzat (BME rektori utasítás a kötelezettségvállalás rendjéről) részletei alapján "
               "válaszolj. Csak a részletekre támaszkodj, magyarul, tömören. A végén zárójelben add meg, melyik "
               "§-ból vagy melléklet-sorból származik a válasz, pl. „(20. § (3))”. A táblázatsorokban a cellák az "
               "„Oszlopok” sorrendjében követik egymást. Ha a válasz nincs a részletekben, írd azt, hogy a dokumentum "
               "ezt nem tartalmazza.\n\nRészletek:\n{context}"),
    ("human", "{question}"),
])

# ----------------------------------------------------------------------------- 1–2. tisztítás és darabolás
NOISE = re.compile(r"^(REKTORI UTASÍTÁS|A KÖTELEZETTSÉGVÁLLALÁS|RENDJÉRŐL|Azonosító:.*|Verziószám:.*|Iktatószám:.*"
                   r"|REKIKT 63/2/2026|\d{1,2}/\d{1,2}\.?|Kötelezettségvállalás rendjéről szóló\s+rektori utasítás"
                   r"|1\. melléklete|Döntési eljárás és kötelezettségvállalási szabályok.*|Ssz\. Kötelezettségvállalás.*"
                   r"|Döntéshozatali folyamatban részt vevők.*|.*\.{6,}.*)$")
# a melléklet oldalainak alján elforgatott kategória-címkék (a pypdf a sor végére fűzi őket)
ANNEX_LABELS = {"Cégalapítás, stratégiai szerződések", "Jogi ügyek", "Szellemi tulajdonjogi ügyek", "IT eszközök",
                "Pénzügy", "Ingatlan", "Pályázatok", "Egyéb, nem nevesített, bevételi pénzügyi kötelezettségvállalások",
                "Oktatási, hallgatói ügyek", "Beszerzések", "Humánerőforrás-menedzsment"}
COLUMNS = ("Ssz. | Kötelezettségvállalás tárgya | Összeghatár | Javaslattevő vagy előzetes véleményező | "
           "Előzetes döntéshozó | Döntéshozó | Kötelezettségvállaló és képviselő | Költséggazda | Szakmai gazda")
ANNEX_FIRST_PAGE = 24


def _clean(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not NOISE.match(ln.strip())]


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_units() -> list[Document]:
    """Egy Document = egy § vagy egy táblázatsor; metadata: section, title."""
    pages = [p.extract_text() or "" for p in PdfReader(PDF).pages]
    units = []

    # törzsszöveg: "17. § [Cím]" sorok mentén
    body = "\n".join("\n".join(_clean(t)) for t in pages[:ANNEX_FIRST_PAGE - 1])
    heads = list(re.finditer(r"^(\d{1,2})\. § \[([^\]]+)\]\s*$", body, re.M))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        text = re.sub(r"\n\d{1,2}\. [A-ZÁÉÍÓÖŐÚÜŰ][^\n§]{3,80}\s*$", "", body[m.end():end].rstrip())  # fejezetcím le
        units.append(Document(page_content=_one_line(text),
                              metadata={"section": f"{m.group(1)}. §", "title": _one_line(m.group(2))}))

    # melléklet: "2.4." sorszámok mentén, minden sor elé az oszlopnevek
    for t in pages[ANNEX_FIRST_PAGE - 1:]:
        lines = _clean(t)
        while lines and lines[-1] in ANNEX_LABELS:
            lines.pop()
        parts = re.split(r"(?:(?<=\n)|(?<=^))(\d{1,2}\.\d{1,2}\.?)\s", "\n".join(lines))
        for k in range(1, len(parts) - 1, 2):
            row_id = parts[k].rstrip(".") + "."
            row, _, notes = _one_line(parts[k + 1]).partition("Megjegyzések:")
            title = re.split(r"\s(?:\d[\d ]*Ft|összeghatártól|nem releváns|≤|≥)", row, maxsplit=1)[0][:120]
            units.append(Document(page_content=f"Oszlopok: {COLUMNS}\nSor {row_id} {row.strip()}",
                                  metadata={"section": f"1. melléklet {row_id}", "title": title}))
            if notes:
                units.append(Document(page_content=notes.strip(),
                                      metadata={"section": "1. melléklet", "title": "Megjegyzések"}))
    return units


def load_chunks() -> list[Document]:
    """A hosszú §-okat tovább vágjuk, és 3. kontextus-dúsítás: minden darab elejére a § és a cím."""
    chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).split_documents(load_units())
    for c in chunks:
        c.page_content = f"[{c.metadata['section']} – {c.metadata['title']}]\n{c.page_content}"
    return chunks


# ----------------------------------------------------------------------------- 4. hibrid keresés
_stemmer = Stemmer.Stemmer("hungarian")
STOP = set("a az egy és vagy is nem hogy ha mi mit ki kit kinek melyik milyen mikor hol hová mennyi mekkora van volt "
           "lesz kell lehet csak már még meg de el ez azt ezt aki ami amely szerint esetén által jelen".split())


def tokenize(text: str) -> list[str]:
    return _stemmer.stemWords([w for w in re.findall(r"\w+", text.lower()) if w not in STOP])


def rrf(rankings: list[list[Document]], top_n: int, k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: pontszám = sum 1 / (k + helyezés) – nem kell a két pontszámskálát összehangolni."""
    score, by_id = {}, {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            key = d.page_content
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank + 1)
            by_id[key] = d
    return [by_id[key] for key in sorted(score, key=score.get, reverse=True)[:top_n]]


def build(top_k: int = 5):
    chunks = load_chunks()
    store = InMemoryVectorStore.from_documents(chunks, embedding=get_embeddings())
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])
    chain = PROMPT | get_llm() | StrOutputParser()

    def retrieve(question: str) -> list[Document]:
        by_vector = store.similarity_search(question, k=10)
        scores = bm25.get_scores(tokenize(question))
        by_keyword = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:10]]
        return rrf([by_vector, by_keyword], top_n=top_k)

    def ask(question: str) -> dict:
        docs = retrieve(question)
        answer = chain.invoke({"context": "\n\n".join(d.page_content for d in docs), "question": question})
        return {"answer": answer, "contexts": [d.page_content for d in docs],
                "sources": [d.metadata["section"] for d in docs]}

    return ask


if __name__ == "__main__":
    main(build, "Jobb RAG")

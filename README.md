# RAG labor

```
git clone https://github.com/hegeduscs/rag_lab && cd rag_lab
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python --version          # 3.11–3.13 kell (a graphrag a 3.14-et még nem támogatja)
pip install -r requirements.txt
```

Futtatás:
```
python naive_rag.py                 # 1. naive RAG (LangChain)
python jobb_rag.py                  # 2. jobb RAG: § szerinti darabolás, hibrid keresés
python eval_rag.py --mutat          # 3. mérés 20 kérdésen – a kész eredmény, LLM-hívás nélkül
python eval_rag.py --mutat q06,q19  #    a változatok válaszai egymás mellett, a bíró indoklásával
python graph_rag.py viz             # 4. GraphRAG: a kész index gráfja böngészőben (graphrag_munka/graf.html)
python graph_rag.py "kérdés"        #    basic / local / global összehasonlítás
```

A `.env` (másold a `.env.example`-t): AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT (gpt-5.4-mini),
AZURE_OPENAI_EMBEDDING_DEPLOYMENT (text-embedding-3-small) – minta: `.env.example`.

A mérés eredménye a repóban van (`eval_eredmenyek.jsonl`, böngészőben: `eval_riport.html`), mert az újramérés
kérdésenként és változatonként két LLM-hívás. Újramérni csak akkor kell, ha változtattál a RAG-on:
`python eval_rag.py --only q06,q07` (a többi tárolt eredmény megmarad), teljes: `python eval_rag.py --graphrag local`.

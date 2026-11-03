"""3. Mérés: ugyanaz a 20 kérdés minden változaton – a számok döntenek, nem a „vibe”.

Az eredmény a repóban van (eval_eredmenyek.jsonl + eval_riport.html), ezért futtatni NEM kell – elég megnézni:

    python eval_rag.py --mutat              # összesítő táblázat, kérdéstípusonként is
    python eval_rag.py --mutat q06,q19      # a változatok válaszai egymás mellett, a bíró indoklásával
    (vagy nyisd meg böngészőben: eval_riport.html)

Újramérés (LLM-hívások: ~20 kérdés × változat × 2 – csak ha változtattál a RAG-on):

    python eval_rag.py                      # naive és jobb RAG
    python eval_rag.py --graphrag local     # + GraphRAG (az index a repóban van)
    python eval_rag.py --only q06,q07       # csak néhány kérdés – a többi tárolt eredmény megmarad

A kérdések szándékosan nehezek: következtetés az összeghatárokból, a melléklet táblázatának sorai, ritka kivételek,
teljes felsorolás és két csapda (nincs benne a dokumentumban) – ezeken válnak szét a változatok.

Kérdésenként két szám:
  hit   – a visszakeresett darabok között ott van-e a bizonyító mondat (ingyenes, determinisztikus;
          a GraphRAG-nál nem mérjük, mert az nem darabokat ad vissza),
  score – LLM-bíró (gpt-5.4-mini) 0 / 0,5 / 1 pontot ad a válaszra a referencia-válasz alapján.
Ugyanez az elv (LLM-as-judge), mint a RAGAS „answer correctness” metrikájánál.
"""
import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "eval_eredmenyek.jsonl"
REPORT = HERE / "eval_riport.html"

QUESTIONS = [
    # --- következtetés az összeghatárokból (a válasz nincs szó szerint a szövegben)
    {"id": "q01", "type": "következtetés", "question": "Ki jegyzi ellen pénzügyileg egy nettó 8 millió forintos beszerzési szerződést?",
     "reference": "A gazdasági és műszaki főigazgató (a Gazdasági Igazgatóság gazdasági vezetőjeként): a pénzügyi ellenjegyzés jogát csak nettó 5 000 000 Ft-ig ruházza át (1 M Ft-ig pénzügyi szakértő, 1–5 M Ft között GI igazgató), e fölött nincs átruházás.",
     "evidence": "nettó 5.000.000 Ft összeghatárig GI igazgató"},
    {"id": "q02", "type": "kivétel", "question": "Ki jegyez ellen pénzügyileg egy nettó 30 millió forintos munkaszerződést?",
     "reference": "A Gazdasági Igazgatóság pénzügyi szakértője: munkaügyi okiratoknál a pénzügyi ellenjegyzés összeghatártól függetlenül a pénzügyi szakértőre van átruházva.",
     "evidence": "munkaügyi okiratok esetén összeghatártól függetlenül"},
    {"id": "q03", "type": "számítás", "question": "Mekkora a kötelezettségvállalás értéke egy határozatlan időre kötött, havi nettó 100 000 forintos szolgáltatási szerződésnél?",
     "reference": "3 600 000 Ft: határozatlan idejű kötelezettségnél az okirat hatálybalépésétől számított 36 hónap kötelezettségeinek összegével kell számolni (36 × 100 000 Ft).",
     "evidence": "harminchat hónapon keresztül"},
    {"id": "q04", "type": "következtetés", "question": "A Rektori Hivatalhoz tartozó egységnél egy nettó 50 millió forintos szerződés teljesítését ki igazolhatja?",
     "reference": "A kötelezettségvállalási jogkör szerinti terület alapján a rektorhelyettes vagy a rektor (10 000 001 és 100 000 000 Ft között).",
     "evidence": "a rektorhelyettes vagy a rektor"},
    {"id": "q05", "type": "kivétel", "question": "A Műszaki Igazgatóságnál egy 800 000 forintos megrendelés teljesítését az igazgatón kívül ki igazolhatja még?",
     "reference": "Az illetékes irodavezető: a Műszaki Igazgatósághoz tartozó kötelezettségvállalásoknál 1 000 000 Ft-ig neki is van teljesítésigazolási joga.",
     "evidence": "irodavezetőnek is van teljesítésigazolási joga"},
    # --- a melléklet táblázata: a sor és az oszlop is számít
    {"id": "q06", "type": "táblázat", "question": "Ki vállal kötelezettséget egy kar 20 millió forintos marketingkampányánál?",
     "reference": "A rektor és a dékán együttesen (kari kommunikációs tevékenység, 1 000 001–100 000 000 Ft; az előzetes döntéshozó a dékán, a döntéshozó a rektor).",
     "evidence": "kari kommunikációs tevékenységhez kapcsolódóan 1 000 001 Ft-tól"},
    {"id": "q07", "type": "táblázat", "question": "Egy 20 millió forintos központi kommunikációs kampánynál ki a döntéshozó és ki a kötelezettségvállaló?",
     "reference": "A döntéshozó a rektor; a kötelezettségvállaló a rektor és a szervezet szakterületileg illetékes második szintű vezetője együttesen (az előzetes döntéshozó a kabinetfőnök és a szakterületileg illetékes második szintű vezető).",
     "evidence": "központi kommunikációs tevékenységhez kapcsolódóan 10 000 001"},
    {"id": "q08", "type": "táblázat", "question": "Ki az előzetes döntéshozó egy 500 ezer forintos IT eszköz beszerzésénél?",
     "reference": "Az IDTI igazgató (a döntéshozó a szervezet szakterületileg illetékes második szintű vezetője, a kötelezettségvállaló a kettő együtt).",
     "evidence": "IT eszközök és szoftverek beszerzése és kapcsolódó szerződések 1 000 000 Ft-ig"},
    {"id": "q09", "type": "táblázat", "question": "Ki dönt egy 150 millió forintos informatikai beszerzésről, és ki írja alá?",
     "reference": "A Rektori Menedzsment Testület dönt; a kötelezettségvállaló a Testület nevében a rektor és a gazdasági és műszaki főigazgató együttesen.",
     "evidence": "IT eszközök és szoftverek beszerzése és kapcsolódó szerződések 100 000 001 Ft-tól"},
    {"id": "q10", "type": "táblázat", "question": "Egy 1,5 millió forintos, egyetlen ajánlat alapján történő beszerzésnél ki az előzetes döntéshozó?",
     "reference": "A pénzügyi szakértő és a beszerzési szakértő (2 000 000 Ft alatti, 1 ajánlatos beszerzés; a döntéshozó a szervezet szakterületileg illetékes második szintű vezetője).",
     "evidence": "(1 ajánlatos beszerzések) 2 000 000 Ft-ig"},
    {"id": "q11", "type": "táblázat", "question": "Hatósági bírság megfizetésénél ki a kötelezettségvállaló?",
     "reference": "A szervezet szakterületileg illetékes második szintű vezetője egyszemélyben (összeghatártól függetlenül; az előzetes döntéshozó a gazdasági igazgató).",
     "evidence": "második szintű vezetője egyszemélyben"},
    {"id": "q12", "type": "táblázat", "question": "Ki az előzetes döntéshozó egy 5 millió forintos bútorvásárlásnál?",
     "reference": "A műszaki igazgató és a gazdasági és műszaki főigazgató (ingatlanhoz kapcsolódó eszközvásárlás 10 M Ft-ig; döntéshozó a szakterületileg illetékes második szintű vezető).",
     "evidence": "lakberendezés stb.) 10 000 000 Ft -ig"},
    # --- kivételek, ritka szabályok
    {"id": "q13", "type": "kivétel", "question": "Kell-e pénzügyi ellenjegyzés egy 150 000 Ft + ÁFA értékű, ajánlat nélküli beszerzéshez?",
     "reference": "Nem: az ajánlat nélküli beszerzés 200 000 Ft + ÁFA összegig kivétel a pénzügyi ellenjegyzés alól.",
     "evidence": "ajánlat nélküli beszerzés (200.000 Ft+ÁFA összegig)"},
    {"id": "q14", "type": "kivétel", "question": "Ha a gazdasági és műszaki főigazgató összeférhetetlen egy pénzügyi ellenjegyzésnél, ki jár el helyette?",
     "reference": "A gazdasági igazgató (és fordítva: a gazdasági igazgató összeférhetetlensége esetén a gazdasági és műszaki főigazgató).",
     "evidence": "összeférhetetlensége esetén a gazdasági igazgató"},
    {"id": "q15", "type": "kivétel", "question": "Továbbadhatja-e valakinek az, akire a rektor átruházta a kötelezettségvállalási jogát?",
     "reference": "Nem: az átruházott jog tovább nem ruházható.",
     "evidence": "Az átruházott jog tovább nem ruházható"},
    {"id": "q16", "type": "értelmezés", "question": "Egyetemi bankkártyás vásárlásnál mi számít egy kötelezettségvállalásnak az értékhatárok szempontjából?",
     "reference": "Az egy alkalommal, egy helyről, egy gazdasági eseményhez kapcsolódóan, egy számla ellenében történő vásárlás.",
     "evidence": "egy számla ellenében történő vásárlás"},
    # --- teljes felsorolás (a top-k könnyen csak egy részét hozza)
    {"id": "q17", "type": "felsorolás", "question": "Sorold fel, milyen esetekben nem kell pénzügyi ellenjegyzés!",
     "reference": "Öt eset: a) hallgatói jogviszony létesítése, megszüntetése, hallgatói státusz okiratai; b) az Egyetem által adott indikatív vagy kötelező érvényű árajánlatok; c) ajánlat nélküli beszerzés 200 000 Ft + ÁFA-ig; d) hatósági határozatok; e) jogszabály alapján kötelezően fizetendő díjak.",
     "evidence": "Pénzügyi ellenjegyzés valamennyi pénzügyi kihatással járó"},
    {"id": "q18", "type": "felsorolás", "question": "Mely szerepkörök nem lehetnek ugyanaz a személy ugyanannál a gazdasági eseménynél?",
     "reference": "Az együttes kötelezettségvállalók egymással (és nem lehetnek egymás hozzátartozói); a kötelezettségvállaló és a pénzügyi ellenjegyző; az utalványozó a kötelezettségvállalóval és a pénzügyi ellenjegyzővel; az érvényesítő a kötelezettségvállalóval, a pénzügyi ellenjegyzővel, az utalványozóval és a teljesítésigazolóval.",
     "evidence": "Az érvényesítő ugyanazon gazdasági esemény tekintetében"},
    # --- csapdák: nincs benne a dokumentumban
    {"id": "q19", "type": "nincs a doksiban", "question": "Ki a BME rektora?",
     "reference": "A dokumentum nem nevezi meg a rektort (csak a szabályzat felelőseit: Riczel Zsuzsanna, Dr. Bauer Dávid, dr. Bíró Barbara). A helyes válasz: nem található a dokumentumban.",
     "evidence": ""},
    {"id": "q20", "type": "nincs a doksiban", "question": "Hány százalékos kötbért kell kikötni egy beszerzési szerződésben?",
     "reference": "A dokumentum ezt nem tartalmazza; csak annyit ír, hogy ha a számla kötbért tartalmaz, a teljesítésigazolásnak ki kell terjednie a kötbér megalapozottságára és mértékére.",
     "evidence": ""},
]

JUDGE_PROMPT = """Egy dokumentum-alapú kérdés-válasz rendszer válaszát értékeled.

Kérdés: {question}
Referencia-válasz (helyes): {reference}
Értékelendő válasz: {answer}

Pontozás:
- 1: a válasz tartalmilag helyes és a lényeget tartalmazza (megfogalmazás, többletinformáció nem számít, ha nem téves),
- 0.5: részben helyes vagy hiányos,
- 0: téves, nem válaszol, vagy kitalált (a referenciának ellentmondó) állítást tartalmaz.
Ha a referencia szerint a dokumentum nem tartalmazza a választ, akkor az 1 pontos válasz az, amelyik ezt mondja.

Csak JSON-t adj vissza: {{"score": 0 | 0.5 | 1, "indoklas": "<egy mondat>"}}"""

QBY = {q["id"]: q for q in QUESTIONS}
TYPES = list(dict.fromkeys(q["type"] for q in QUESTIONS))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------- mérés (LLM-hívások)

def evaluate(name: str, ask, questions: list[dict], judge, stamp: str) -> list[dict]:
    print(f"\n== {name}")
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        r = ask(q["question"])
        latency = time.perf_counter() - t0
        hit = None
        if q["evidence"] and r.get("contexts") is not None:
            hit = any(norm(q["evidence"]) in norm(c) for c in r["contexts"])
        verdict = judge.invoke({"question": q["question"], "reference": q["reference"], "answer": r["answer"]})
        row = {"variant": name, "id": q["id"], "type": q["type"], "question": q["question"], "answer": r["answer"],
               "sources": r.get("sources"), "hit": hit, "score": float(verdict.get("score", 0)),
               "indoklas": verdict.get("indoklas", ""), "latency_s": round(latency, 1), "datum": stamp}
        rows.append(row)
        print(f"  {q['id']} {q['type']:<16} hit={'-' if hit is None else int(hit)}  score={row['score']:<3}  "
              f"{latency:4.1f}s")
    return rows


def run(args) -> None:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    import jobb_rag
    import naive_rag

    questions = [q for q in QUESTIONS if not args.only or q["id"] in args.only.split(",")]
    judge = ChatPromptTemplate.from_template(JUDGE_PROMPT) | naive_rag.get_llm() | JsonOutputParser()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    new = []
    for name, build in [("naive", naive_rag.build), ("jobb", jobb_rag.build)]:
        print(f"\n{name}: index építése...", flush=True)
        new += evaluate(name, build(), questions, judge, stamp)
    if args.graphrag:
        import graph_rag
        new += evaluate(f"graphrag-{args.graphrag}",
                        lambda q: {"answer": graph_rag.ask(q, args.graphrag), "contexts": None},
                        questions, judge, stamp)

    # összefésülés a tárolt eredménnyel: az újramért (változat, kérdés) párok felülírják a régit, a többi marad
    fresh = {(r["variant"], r["id"]) for r in new}
    rows = [r for r in load() if (r["variant"], r["id"]) not in fresh] + new
    save(rows)
    print_summary(rows)
    print(f"\nMentve: {RESULTS.name} és {REPORT.name} (ezeket érdemes a repóba tenni)")


# ---------------------------------------------------------------- tárolt eredmény

def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(rows: list[dict]) -> None:
    order = {q["id"]: i for i, q in enumerate(QUESTIONS)}
    rows = sorted(rows, key=lambda r: (variant_order(rows).index(r["variant"]), order.get(r["id"], 99)))
    with RESULTS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT.write_text(render_html(rows), encoding="utf-8")


def variant_order(rows: list[dict]) -> list[str]:
    names = list(dict.fromkeys(r["variant"] for r in rows))
    fixed = [v for v in ("naive", "jobb") if v in names]
    return fixed + sorted(v for v in names if v not in fixed)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for v in variant_order(rows):
        rs = [r for r in rows if r["variant"] == v]
        hits = [r["hit"] for r in rs if r["hit"] is not None]
        by_type = {}
        for t in TYPES:
            ts = [r["score"] for r in rs if r["type"] == t]
            by_type[t] = sum(ts) / len(ts) if ts else None
        out.append({"variant": v, "n": len(rs), "hit_rate": sum(hits) / len(hits) if hits else None,
                    "score": sum(r["score"] for r in rs) / len(rs),
                    "latency": sum(r["latency_s"] for r in rs) / len(rs),
                    "by_type": by_type, "datum": max((r.get("datum") or "") for r in rs)})
    return out


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def print_summary(rows: list[dict]) -> None:
    summ = summarize(rows)
    print(f"\n{'változat':<16}{'kérdés':>7}{'hit rate':>10}{'score':>8}{'latency':>9}   mérve")
    for s in summ:
        print(f"{s['variant']:<16}{s['n']:>7}{fmt(s['hit_rate']):>10}{fmt(s['score']):>8}"
              f"{fmt(s['latency'], 1):>8}s   {s['datum'] or '?'}")
    print(f"\nscore kérdéstípusonként\n{'típus':<22}" + "".join(f"{s['variant']:>16}" for s in summ))
    for t in TYPES:
        n = sum(1 for q in QUESTIONS if q["type"] == t)
        print(f"{t + f' ({n})':<22}" + "".join(f"{fmt(s['by_type'][t]):>16}" for s in summ))


def show(rows: list[dict], ids: str | None) -> None:
    if not rows:
        sys.exit(f"Még nincs tárolt eredmény ({RESULTS.name}). Az oktató futtatja egyszer: python eval_rag.py --graphrag local")
    print_summary(rows)
    if not ids:
        print("\nEgy kérdés részletei: python eval_rag.py --mutat q06   (kérdések: " +
              ", ".join(f"{q['id']} {q['type']}" for q in QUESTIONS[:3]) + ", …)")
        return
    for qid in ids.split(","):
        q = QBY.get(qid.strip())
        if not q:
            print(f"\nNincs ilyen kérdés: {qid}")
            continue
        print(f"\n{'=' * 100}\n{q['id']} [{q['type']}] {q['question']}\n✅ referencia: {q['reference']}")
        for r in [r for r in rows if r["id"] == q["id"]]:
            hit = "" if r["hit"] is None else f"  hit={int(r['hit'])}"
            print(f"\n--- {r['variant']}: score={r['score']}{hit}   📎 {', '.join(r.get('sources') or []) or '-'}")
            print(r["answer"].strip())
            print(f"⚖️  bíró: {r['indoklas']}")


# ---------------------------------------------------------------- HTML riport (mutogatáshoz)

CSS = """
:root{--bg:#fff;--fg:#1d2433;--mut:#5b6475;--line:#dde2ea;--card:#f5f7fa;--acc:#0a6fc2;
--s1:#d9f2e0;--s1f:#14532d;--s5:#fdf0c7;--s5f:#713f12;--s0:#fbdcdc;--s0f:#7f1d1d}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2c323d;--card:#1c2028;
--acc:#5aa9ec;--s1:#173d26;--s1f:#bff0cf;--s5:#3d3313;--s5f:#f6e3a1;--s0:#46201f;--s0f:#f7c6c4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 10px}.mut{color:var(--mut)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card b{font-size:15px}.big{font-size:32px;font-weight:700;color:var(--acc);line-height:1.2}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{font-size:13px;color:var(--mut);font-weight:600}td.c,th.c{text-align:center;white-space:nowrap}
.s1{background:var(--s1);color:var(--s1f)}.s05{background:var(--s5);color:var(--s5f)}.s0{background:var(--s0);color:var(--s0f)}
.pill{display:inline-block;min-width:36px;padding:1px 8px;border-radius:999px;font-weight:600;font-size:13px}
details{border:1px solid var(--line);border-radius:10px;margin:8px 0;background:var(--card)}
summary{cursor:pointer;padding:10px 12px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
summary .q{flex:1 1 300px}.tag{font-size:12px;color:var(--mut);white-space:nowrap}
.body{padding:0 12px 12px}.ref{border-left:3px solid var(--acc);padding:6px 10px;margin:4px 0 10px}
.ans{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.ans>div{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;white-space:pre-wrap;font-size:14px}
.ans .h{font-weight:600;margin-bottom:4px;white-space:normal}.jud{color:var(--mut);font-size:13px;margin-top:6px;white-space:normal}
"""


def pill(x, avg: bool = False) -> str:
    """Színes címke: egy kérdés pontja (0 / ½ / 1) vagy átlag (két tizedes)."""
    if x is None:
        return '<span class="mut">-</span>'
    cls = "s1" if x >= 0.85 else "s05" if x >= 0.45 else "s0"
    label = f"{x:.2f}" if avg else {1.0: "1", 0.5: "½", 0.0: "0"}.get(float(x), f"{x:g}")
    return f'<span class="pill {cls}">{label}</span>'


def render_html(rows: list[dict]) -> str:
    e = html.escape
    summ = summarize(rows)
    vs = [s["variant"] for s in summ]
    cards = "".join(
        f'<div class="card"><b>{e(s["variant"])}</b><div class="big">{s["score"]:.2f}</div>'
        f'<div class="mut">score · hit rate {fmt(s["hit_rate"])} · {fmt(s["latency"], 1)} s/kérdés</div></div>'
        for s in summ)
    head = "".join(f'<th class="c">{e(v)}</th>' for v in vs)
    by_type = "".join(
        f'<tr><td>{e(t)} <span class="mut">({sum(1 for q in QUESTIONS if q["type"] == t)})</span></td>'
        + "".join(f'<td class="c">{pill(s["by_type"][t], avg=True)}</td>' for s in summ) + "</tr>" for t in TYPES)
    cell = {(r["variant"], r["id"]): r for r in rows}
    items = []
    for q in QUESTIONS:
        rs = [cell.get((v, q["id"])) for v in vs]
        if not any(rs):
            continue
        scores = "".join(f'<span class="tag">{e(v)}</span> {pill(r["score"] if r else None)}' for v, r in zip(vs, rs))
        answers = "".join(
            f'<div><div class="h">{e(v)} {pill(r["score"])}'
            f'{"" if r["hit"] is None else " · hit " + str(int(r["hit"]))}'
            f'{"" if not r.get("sources") else " <span class=mut>· " + e(", ".join(r["sources"])) + "</span>"}'
            f'</div>{e(r["answer"].strip())}'
            f'<div class="jud">⚖️ {e(r["indoklas"])}</div></div>'
            for v, r in zip(vs, rs) if r)
        items.append(f'<details><summary><b>{q["id"]}</b><span class="q">{e(q["question"])}</span>'
                     f'<span class="tag">{e(q["type"])}</span>{scores}</summary><div class="body">'
                     f'<div class="ref"><b>Referencia:</b> {e(q["reference"])}</div><div class="ans">{answers}</div>'
                     f'</div></details>')
    datum = max((s["datum"] for s in summ), default="")
    return f"""<!doctype html><html lang="hu"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG eval riport</title><style>{CSS}</style></head>
<body><main><h1>RAG eval – {len(QUESTIONS)} nehéz kérdés</h1>
<div class="mut">BME kötelezettségvállalási rend · LLM-bíró: 0 / ½ / 1 pont · mérve: {e(datum or "?")}</div>
<div class="cards">{cards}</div>
<h2>Score kérdéstípusonként</h2><div class="wrap"><table><tr><th>típus</th>{head}</tr>{by_type}</table></div>
<h2>Kérdésenként <span class="mut">– kattints a válaszokért és a bíró indoklásáért</span></h2>
{''.join(items)}
<p class="mut">hit = a visszakeresett darabokban ott van-e a bizonyító mondat (a GraphRAG-nál nem mérjük) ·
újragenerálás: <code>python eval_rag.py --riport</code></p></main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutat", nargs="?", const="", metavar="KÉRDÉSEK",
                    help="a tárolt eredményt mutatja, LLM-hívás nélkül; opcionálisan kérdés-azonosítók, pl. q06,q19")
    ap.add_argument("--riport", action="store_true", help="az eval_riport.html újragenerálása a tárolt eredményből")
    ap.add_argument("--only", help="újramérés csak ezekre a kérdésekre, pl. q01,q15")
    ap.add_argument("--graphrag", metavar="MÓD", help="a GraphRAG-ot is méri: local | global | basic | drift")
    args = ap.parse_args()
    if args.mutat is not None:
        show(load(), args.mutat or None)
    elif args.riport:
        rows = load()
        if not rows:
            sys.exit(f"Nincs tárolt eredmény: {RESULTS.name}")
        REPORT.write_text(render_html(rows), encoding="utf-8")
        print(f"Kész: {REPORT.name}")
    else:
        run(args)


if __name__ == "__main__":
    main()

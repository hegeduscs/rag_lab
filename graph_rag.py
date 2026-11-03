"""3. GraphRAG (Microsoft graphrag 3.3): entitások és kapcsolatok gráfja + közösségi riportok.

    python graph_rag.py index                       # oktatók, egyszer: munkamappa + indexelés (pár perc, ~1 USD)
    python graph_rag.py "Mik a szabályzat legfontosabb témái?"                  # basic / local / global egymás után
    python graph_rag.py viz                         # graphrag_munka/graf.html: a gráf böngészőben
    python graph_rag.py --method local "Ki jegyzi ellen pénzügyileg egy nettó 3 millió forintos kötelezettségvállalást?"

Keresési módok:
  basic  – sima vektoros keresés a szövegdarabokon (= naive RAG, összehasonlítási alap)
  local  – a kérdéshez illő entitásokból indul, hozza a gráfszomszédokat és a hozzájuk tartozó szövegeket
  global – map-reduce a közösségi riportokon: „miről szól az egész?” típusú kérdésekre (sok LLM-hívás)
  drift  – local + global, kérdés-finomítással (a legdrágább)

Az indexelés a graphrag_munka/ mappába dolgozik: a `graphrag init` alapbeállításait itt igazítjuk Azure-ra és a
magyar dokumentumhoz (lásd prepare()); a settings.yaml és a prompts/ utána kézzel is szerkeszthető.

A kész index (graphrag_munka/output) a repóban van, így a résztvevőknek nem kell indexelniük, csak kérdezni.
Az LLM-válaszok gyorsítótára (graphrag_munka/cache) nincs a repóban: aki újraindexel (--force), az újra fizet (~1 USD);
az oktatói gépen a gyorsítótár miatt ez majdnem ingyenes, amíg a promptok és a beállítások nem változnak.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
WORK = HERE / "graphrag_munka"
load_dotenv(HERE / ".env")
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
# a settings.yaml ${...} változói – ha a .env-ből hiányoznak, ésszerű alapértékek
ENV.setdefault("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
ENV.setdefault("AZURE_OPENAI_CHAT_MODEL", ENV.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini"))
ENV.setdefault("AZURE_OPENAI_EMBEDDING_MODEL", ENV.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"))

AZURE_MODELS = """completion_models:
  default_completion_model:
    model_provider: azure
    model: ${AZURE_OPENAI_CHAT_MODEL}
    azure_deployment_name: ${AZURE_OPENAI_CHAT_DEPLOYMENT}
    api_base: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}
    auth_method: api_key
    api_key: ${AZURE_OPENAI_API_KEY}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: azure
    model: ${AZURE_OPENAI_EMBEDDING_MODEL}
    azure_deployment_name: ${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}
    api_base: ${AZURE_OPENAI_ENDPOINT}
    api_version: ${AZURE_OPENAI_API_VERSION}
    auth_method: api_key
    api_key: ${AZURE_OPENAI_API_KEY}
    retry:
      type: exponential_backoff

"""
ENTITY_TYPES = "[szervezeti egység, tisztség, szerepkör, testület, okirat, folyamat, összeghatár, jogszabály]"


def graphrag(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "graphrag", *args], env=ENV, capture_output=capture,
                          text=True, encoding="utf-8", errors="replace")


def prepare():
    """graphrag init + az alapbeállítások átírása Azure-ra és magyar szövegre + a bemeneti szöveg."""
    if not (WORK / "settings.yaml").exists():
        r = graphrag("init", "--root", str(WORK), "-m", ENV["AZURE_OPENAI_CHAT_MODEL"],
                     "-e", ENV["AZURE_OPENAI_EMBEDDING_MODEL"])
        if r.returncode:
            sys.exit("graphrag init hiba")
        (WORK / ".env").unlink(missing_ok=True)  # a kulcsok a labor .env-jéből jönnek

        s = (WORK / "settings.yaml").read_text(encoding="utf-8")
        # 1) modellek: Azure OpenAI, a .env változóival
        s = re.sub(r"completion_models:.*?(?=### Document processing)", lambda _: AZURE_MODELS, s, flags=re.S)
        # 2) a dokumentumhoz illő entitástípusok (alapból: organization, person, geo, event)
        s = re.sub(r"entity_types: \[.*?\]", "entity_types: " + ENTITY_TYPES, s)
        # 3) DRIFT: alapból temperature=0-t küld, a gpt-5.x reasoning mellett csak 1-et fogad el;
        #    az init nem létező reduce-prompt fájlra hivatkozik
        s = s.replace('reduce_prompt: "prompts/drift_search_reduce_prompt.txt"',
                      'reduce_prompt: "prompts/drift_reduce_prompt.txt"\n  reduce_temperature: 1\n'
                      '  local_search_temperature: 1')
        # 4) a gráf GraphML-ben is (output/graph.graphml) – Gephivel megnyitható
        s = s.replace("graphml: false", "graphml: true")
        (WORK / "settings.yaml").write_text(s, encoding="utf-8")

        # 5) promptok: a kinyerés alapból angolul írná le a magyar entitásokat
        p = WORK / "prompts"
        _edit(p / "extract_graph.txt", "Return output in English", "Return output in Hungarian")
        for f in ("community_report_graph.txt", "community_report_text.txt"):
            _edit(p / f, "Return output as a well-formed JSON",
                  "Write all text values of the report in Hungarian.\nReturn output as a well-formed JSON")
        for f in ("local_search_system_prompt.txt", "global_search_reduce_system_prompt.txt",
                  "global_search_map_system_prompt.txt", "basic_search_system_prompt.txt",
                  "drift_search_system_prompt.txt", "drift_reduce_prompt.txt"):
            t = (p / f).read_text(encoding="utf-8").rstrip()
            (p / f).write_text(t + "\n\nAnswer in the language of the user's question.\n", encoding="utf-8")

    # 6) bemenet: a jobb_rag.py tisztított, § és táblázatsor szerinti szövege
    from jobb_rag import load_units
    text = "\n\n".join(f"{d.metadata['section']} – {d.metadata['title']}\n{d.page_content}" for d in load_units())
    (WORK / "input").mkdir(parents=True, exist_ok=True)
    (WORK / "input" / "bme_kotelezettsegvallalas.txt").write_text(
        "BME rektori utasítás 2/2026. – A kötelezettségvállalás rendjéről\n\n" + text, encoding="utf-8")


def _edit(path: Path, old: str, new: str):
    t = path.read_text(encoding="utf-8")
    path.write_text(t.replace(old, new, 1), encoding="utf-8")


# ----------------------------------------------------------------------------- vizualizáció
VIZ_HTML = r"""<!doctype html><html lang="hu"><head><meta charset="utf-8"><title>GraphRAG – tudásgráf</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
 body{margin:0;font:14px/1.4 system-ui,sans-serif;display:flex;height:100vh;color:#263238}
 #net{flex:1;background:#fafbfc} #side{width:380px;overflow:auto;border-left:1px solid #ddd;padding:12px 16px}
 h1{font-size:18px;color:#0053A3;margin:4px 0} h2{font-size:15px;margin:14px 0 4px;color:#0091EA}
 .rep{border:1px solid #e3e7ea;border-radius:8px;padding:8px 10px;margin:8px 0;cursor:pointer}
 .rep:hover{border-color:#0091EA} .rep b{color:#0053A3} .small{color:#607d8b;font-size:12px}
 input{width:100%;padding:6px;box-sizing:border-box;margin:6px 0} #info{white-space:pre-wrap}
</style></head><body><div id="net"></div><div id="side">
<h1>Tudásgráf</h1><div class="small">__STATS__ · szín = közösség (0. szint), méret = fokszám</div>
<input id="q" placeholder="Entitás keresése…"><div id="info" class="small">Kattints egy csúcsra vagy egy riportra.</div>
<h2>Közösségi riportok</h2><div id="reps"></div></div>
<script>
const D = __DATA__;
const nodes = new vis.DataSet(D.nodes), edges = new vis.DataSet(D.edges);
const net = new vis.Network(document.getElementById("net"), {nodes, edges}, {
  nodes:{shape:"dot", font:{size:14}}, edges:{color:{color:"#b0bec5", highlight:"#0053A3"}, smooth:false},
  physics:{solver:"forceAtlas2Based", stabilization:{iterations:300}}, interaction:{hover:true}});
const info = document.getElementById("info");
net.on("click", p => {
  if (p.nodes.length) { const n = nodes.get(p.nodes[0]); info.textContent = n.label + "  [" + n.type + "]\n\n" + n.desc; }
  else if (p.edges.length) { const e = edges.get(p.edges[0]); info.textContent = e.from + " → " + e.to + "\n\n" + e.desc; }
});
document.getElementById("q").addEventListener("input", ev => {
  const t = ev.target.value.toLowerCase(); if (!t) return;
  const hit = D.nodes.filter(n => n.label.toLowerCase().includes(t)).map(n => n.id);
  net.selectNodes(hit); if (hit.length) net.focus(hit[0], {scale:1.2, animation:true});
});
const reps = document.getElementById("reps");
for (const r of D.reports) {
  const d = document.createElement("div"); d.className = "rep";
  d.innerHTML = "<b></b><div class=small></div><div></div>";
  d.children[0].textContent = r.title; d.children[1].textContent = r.level + ". szint · " + r.size + " entitás · fontosság " + r.rating;
  d.children[2].textContent = r.summary;
  d.onclick = () => { net.selectNodes(r.nodes); net.fit({nodes:r.nodes, animation:true});
    info.textContent = r.title + "\n\n" + r.summary + "\n\n" + r.findings.map(f => "• " + f).join("\n"); };
  reps.appendChild(d);
}
</script></body></html>"""
PALETTE = ["#0091EA", "#F57C00", "#43A047", "#8E24AA", "#E53935", "#00897B", "#6D4C41", "#3949AB", "#C0CA33",
           "#D81B60", "#5E35B1", "#00ACC1", "#7CB342", "#FB8C00", "#546E7A"]


def visualize() -> Path:
    """Interaktív HTML a felépített gráfról: entitások, kapcsolatok, közösségek és a közösségi riportok."""
    import json

    import pandas as pd

    out = WORK / "output"
    ents = pd.read_parquet(out / "entities.parquet")
    rels = pd.read_parquet(out / "relationships.parquet")
    comms = pd.read_parquet(out / "communities.parquet")
    reports = pd.read_parquet(out / "community_reports.parquet")

    color = {}  # entitás-id -> a 0. szintű közössége színe
    for _, c in comms[comms["level"] == 0].iterrows():
        for eid in c["entity_ids"]:
            color[eid] = PALETTE[int(c["community"]) % len(PALETTE)]
    nodes = [{"id": e["title"], "label": e["title"], "type": e["type"], "desc": e["description"] or "",
              "value": int(e["degree"]), "color": color.get(e["id"], "#90a4ae"),
              "title": f"{e['title']} [{e['type']}]"} for _, e in ents.iterrows()]
    edges = [{"from": r["source"], "to": r["target"], "desc": r["description"] or "", "value": float(r["weight"]),
              "title": r["description"] or ""} for _, r in rels.iterrows()]
    by_comm = {int(c["community"]): c for _, c in comms.iterrows()}
    id2title = dict(zip(ents["id"], ents["title"]))
    reps = []
    for _, r in reports.sort_values(["level", "rank"], ascending=[True, False]).iterrows():
        c = by_comm.get(int(r["community"]))
        reps.append({"title": r["title"], "summary": r["summary"], "level": int(r["level"]),
                     "rating": float(r["rank"]), "size": int(r["size"]),
                     "findings": [f.get("summary", "") for f in r["findings"]],
                     "nodes": [id2title[i] for i in (c["entity_ids"] if c is not None else []) if i in id2title]})
    stats = f"{len(nodes)} entitás, {len(edges)} kapcsolat, {len(reports)} közösségi riport"
    data = json.dumps({"nodes": nodes, "edges": edges, "reports": reps}, ensure_ascii=False).replace("</", "<\\/")
    path = WORK / "graf.html"
    path.write_text(VIZ_HTML.replace("__DATA__", data).replace("__STATS__", stats), encoding="utf-8")
    return path


def ask(question: str, method: str = "local") -> str:
    r = graphrag("query", "--root", str(WORK), "--method", method, "--response-type",
                 "Rövid válasz magyarul, legfeljebb 5 mondat", question, capture=True)
    return r.stdout.strip() if r.returncode == 0 else "HIBA: " + r.stderr.strip()[-600:]


def main():
    ap = argparse.ArgumentParser(description="GraphRAG a BME-s dokumentumon")
    ap.add_argument("question", nargs="*", help='"index", "viz", vagy egy kérdés')
    ap.add_argument("--method", help="basic | local | global | drift (alap: basic, local és global is)")
    ap.add_argument("--force", action="store_true", help="index esetén: a meglévő index újraépítése")
    args = ap.parse_args()

    if args.question == ["index"]:
        if (WORK / "output" / "entities.parquet").exists() and not args.force:
            sys.exit("Már van kész index (graphrag_munka/output) – kérdezni lehet. Újraépítés: python graph_rag.py index --force")
        prepare()
        sys.exit(graphrag("index", "--root", str(WORK)).returncode)
    if not (WORK / "output" / "entities.parquet").exists():
        sys.exit("Még nincs index: python graph_rag.py index")
    if args.question == ["viz"]:
        print(f"Kész: {visualize()} – nyisd meg böngészőben (a GraphML: graphrag_munka/output/graph.graphml, Gephihez)")
        return
    question = " ".join(args.question) or "Mik a szabályzat legfontosabb témái és szereplői?"
    print(f"❓ {question}")
    for m in [args.method] if args.method else ["basic", "local", "global"]:
        print(f"\n--- {m}\n{ask(question, m)}")


if __name__ == "__main__":
    main()

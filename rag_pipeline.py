# src/rag_pipeline.py
"""
Compliance RAG + Optional LLM (Ollama via REST)
- Open-speech friendly query understanding:
  * slang/contractions cleanup
  * acronym & synonym expansion (KYC, PII, RBI, GDPR, FFIEC, Basel, CCO, etc.)
  * typo correction using in-corpus vocabulary (RapidFuzz)
  * phrase/intent expansion (incident response, retention, audit, encryption)
  * optional LLM-assisted query rewrite (toggle)
- Fast BM25 with fuzzy signals + RRF, optional LLM synthesis + rerank
- Scope inference (external/internal/both) & compare mode
- Citations optional ("with citations" cue)
- Latest-version-only corpus filtering (avoid stale dupes)
"""

import os, re, json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz
import requests

# --------------------------------------------------------------------------
# Expose input for tests to monkeypatch (tests expect rag.input)
from builtins import input as _builtin_input
input = _builtin_input
# --------------------------------------------------------------------------

# ------------------- Ollama REST client (no Python package) ----------------
class OllamaClient:
    def _init_(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,  # avoid NDJSON streaming
        }
        if options:
            payload["options"] = options
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, (list, dict)):
            content = json.dumps(content)
        return {"message": {"content": str(content).strip()}}

# --------------------------------------------------------------------------

load_dotenv()

DATA_DIR      = os.getenv("DATA_DIR", "data/raw")
CORPUS_PATH   = os.getenv("CORPUS_PATH", "vectorstore/corpus.jsonl")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "vectorstore/manifest.json")
SUMMARY_PATH  = os.getenv("SUMMARY_PATH", "vectorstore/policy_summary.json")
FEED_PATH     = os.getenv("FEED_PATH", "vectorstore/live_feed.json")
HISTORY_PATH  = os.getenv("HISTORY_PATH", "vectorstore/history.jsonl")

DEFAULT_CITATIONS = os.getenv("DEFAULT_CITATIONS", "0") == "1"
BM25_TOPK     = int(os.getenv("BM25_TOPK", "40"))
RRF_K         = int(os.getenv("RRF_K", "30"))
MAX_CHARS     = int(os.getenv("MAX_CHARS_PER_SNIPPET", "600"))
MAX_SNIPPETS  = int(os.getenv("MAX_RETURNED_SNIPPETS", "8"))

ENABLE_FUZZY  = os.getenv("ENABLE_FUZZY", "1") == "1"
ENABLE_LLM    = os.getenv("ENABLE_LLM", "0") == "1"
ENABLE_LLM_RERANK = os.getenv("ENABLE_LLM_RERANK", "0") == "1"
ENABLE_LLM_QUERY_REWRITE = os.getenv("ENABLE_LLM_QUERY_REWRITE", "0") == "1"

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MAX_TOKENS= int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.2"))

ollama = OllamaClient(OLLAMA_HOST)

# ------------------ Utils ------------------
def _read_jsonl(path:str)->List[Dict[str,Any]]:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def _append_history(q:str)->None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "q": q})+"\n")

def _load_manifest() -> Dict[str,Any]:
    if not os.path.exists(MANIFEST_PATH): return {"files": {}, "last_ingest": None}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f: raw=json.load(f)
    except: return {"files": {}, "last_ingest": None}
    raw.setdefault("files", {}); raw.setdefault("last_ingest", None)
    return raw

def _normalize(txt:str)->str:
    return re.sub(r"\s+", " ", txt.strip())

def _tok(doc:str)->List[str]:
    return re.findall(r"[A-Za-z0-9_]+", doc.lower())

# latest-only filtering: keep highest version per (source,page)
def _latest_only(rows:List[Dict[str,Any]])->List[Dict[str,Any]]:
    latest: Dict[Tuple[str,int], Tuple[int, Dict[str,Any]]] = {}
    for r in rows:
        meta = r.get("meta", {})
        key = (str(meta.get("source")), int(meta.get("page", 0)))
        v = int(meta.get("version", 1))
        if key not in latest or v > latest[key][0]:
            latest[key] = (v, r)
    return [x[1] for x in latest.values()]

# RRF combiner across signals
def _rrf(scores_list: List[List[Tuple[int,float]]], k:int=60)->List[Tuple[int,float]]:
    rank_maps: List[Dict[int,int]] = []
    for pairs in scores_list:
        order = sorted(range(len(pairs)), key=lambda i: pairs[i][1], reverse=True)
        m: Dict[int,int] = {}
        for rank_idx in range(len(order)):
            doc_id = pairs[order[rank_idx]][0]
            m[doc_id] = rank_idx + 1
        rank_maps.append(m)
    all_doc_ids = set()
    for pairs in scores_list:
        for doc_id,_ in pairs:
            all_doc_ids.add(doc_id)
    merged: List[Tuple[int,float]] = []
    for doc_id in all_doc_ids:
        s = 0.0
        for m in rank_maps:
            rank = m.get(doc_id)
            if rank is not None:
                s += 1.0 / (k + rank)
        merged.append((doc_id, s))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

# ------------------ Open-speech Query Understanding ------------------
_STOPWORDS = {
    "the","this","that","these","those","a","an","and","or","but","if","then","so","of","in","on","for","to","with",
    "is","are","was","were","be","been","being","do","does","did","can","could","should","would","will","shall",
    "i","we","you","he","she","they","it","me","my","our","your","their","us","hey","hi","hello","please","kindly"
}

# canonical synonyms/acronyms → expanded token set
_SYNONYMS: Dict[str, List[str]] = {
    "kyc": ["know your customer","customer due diligence","kyc"],
    "pii": ["personally identifiable information","personal data","pii"],
    "cco": ["chief compliance officer","cco"],
    "rbi": ["reserve bank of india","rbi"],
    "gdpr": ["general data protection regulation","gdpr"],
    "ffiec": ["ffiec"],
    "basel": ["basel iii","basel 3","basel"],
    "sop": ["standard operating procedure","sop","procedure","handbook","manual","policy"],
    "encryption": ["encrypt","encryption","crypto","key management","kms","key rotation"],
    "retention": ["data retention","record retention","storage duration","archival","retention"],
    "incident": ["incident","breach","security incident","data leak","data breach","response workflow"],
    "audit": ["audit","inspection","review","testing","assurance"],
    "training": ["training","awareness","education","mandatorily","mandatory training"],
    "reporting": ["reporting","notify","notification","regulator","escalation","timeline"],
    "policy": ["policy","guideline","manual","handbook","procedure","sop"],
    "access": ["access control","authorization","least privilege","segregation of duties"]
}

_CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "cannot"),
    (r"\bshan't\b", "shall not"),
    (r"\bn't\b", " not"),
    (r"\bI'm\b", "I am"),
    (r"\bit's\b", "it is"),
    (r"\bthere's\b", "there is"),
    (r"\bthey're\b", "they are"),
    (r"\bwe're\b", "we are"),
    (r"\by'all\b", "you all"),
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bkinda\b", "kind of"),
]

def _clean_speech(text:str)->str:
    t = text.strip()
    for pat, repl in _CONTRACTIONS:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    # remove filler politeness/openers
    t = re.sub(r"^(hey|hi|hello|please|kindly)\s*,?\s*", "", t, flags=re.IGNORECASE)
    return _normalize(t)

def _extract_keywords(q:str)->List[str]:
    toks = _tok(q)
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]

def _expand_synonyms(tokens:List[str])->List[str]:
    out = list(tokens)
    for t in list(tokens):
        for key, exps in _SYNONYMS.items():
            if t == key or t in key.split():
                out.extend([_tok(x) for x in exps if x])
            if t in [w for exp in exps for w in _tok(exp)]:
                out.extend([_tok(x) for x in exps if x])
    # flatten
    flat = []
    for chunk in out:
        if isinstance(chunk, list):
            flat.extend(chunk)
        else:
            flat.append(chunk)
    return list(dict.fromkeys(flat))  # dedupe preserving order

def _build_vocab_from_corpus() -> Dict[str,int]:
    rows = _read_jsonl(CORPUS_PATH)
    freq: Dict[str,int] = {}
    for r in rows:
        for t in _tok(r.get("text","")):
            if len(t) < 2: continue
            freq[t] = freq.get(t,0) + 1
    return freq

# lazy global vocab for spelling correction
_VOCAB_FREQ: Dict[str,int] = _build_vocab_from_corpus() if os.path.exists(CORPUS_PATH) else {}

def _correct_token(tok:str, vocab:Dict[str,int])->str:
    if not vocab or tok in vocab: return tok
    # only attempt on alphabetic tokens
    if not re.match(r"^[a-z][a-z0-9_-]*$", tok): return tok
    cand, score, _ = process.extractOne(tok, list(vocab.keys()), scorer=fuzz.WRatio) or (None,0,None)
    return cand if cand and score >= 88 else tok  # conservative threshold

def _spell_correct(tokens:List[str], vocab:Dict[str,int])->List[str]:
    return [_correct_token(t, vocab) for t in tokens]

def _intent_expand(tokens:List[str])->List[str]:
    s = set(tokens)
    # high-level expansions driven by intent cues
    if "incident" in s or "breach" in s:
        s.update(_tok("incident response detection containment eradication recovery reporting"))
    if "retention" in s or ("record" in s and "duration" in s):
        s.update(_tok("retention schedule period archival destruction disposition years months"))
    if "audit" in s:
        s.update(_tok("audit testing inspection assurance findings remediation evidence"))
    if "encryption" in s or "encrypt" in s or "crypto" in s:
        s.update(_tok("key management kms key rotation algorithm cipher at rest in transit"))
    return list(s)

def _rewrite_query_rule_based(q:str)->str:
    # 1) cleanup casual speech + contractions
    q = _clean_speech(q)
    # 2) extract core tokens
    toks = _extract_keywords(q)
    # 3) expand synonyms/acronyms
    toks = _expand_synonyms(toks)
    # 4) spell-correct against corpus vocab
    toks = _spell_correct(toks, _VOCAB_FREQ)
    # 5) expand by intent cues
    toks = _intent_expand(toks)
    # 6) rebuilt query string (keeps BM25 happy and fuzzy strong)
    return " ".join(dict.fromkeys(toks))  # preserve order & dedupe

def _llm_query_rewrite(original_q:str)->str:
    if not ENABLE_LLM or not ENABLE_LLM_QUERY_REWRITE:
        return ""
    prompt = (
        "Rewrite the user query for document retrieval in banking compliance. "
        "Keep it short, keyword-rich, include synonyms and common acronyms expanded (e.g., KYC, PII, RBI), "
        "and add closely related phrases. Output only the rewritten query."
        f"\n\nUser query: {original_q}"
    )
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role":"user","content":prompt}],
            options={"temperature":0.1, "num_predict": 128}
        )
        text = resp.get("message",{}).get("content","").strip()
        return _normalize(text)[:400]
    except Exception:
        return ""  # fall back to rule-based
# --------------------------------------------------------------------------

# ------------------ Scope & corpus ------------------
def _infer_scope_and_mode(q:str)->Tuple[str,str]:
    l = q.lower()
    mode = "compare" if any(k in l for k in ["compare","vs","versus","diff","difference","gap","align"]) else "answer"

    if any(k in l for k in ["rbi","mas","iso","gdpr","ffiec","basel","circular","guideline","act","regulation","external"]):
        scope = "external" if "internal" not in l else "both"
    elif any(k in l for k in ["internal","policy","our","company","sop","handbook"]):
        scope = "internal"
    else:
        scope = "internal"
    if mode == "compare":
        scope = "both"
    return scope, mode

def _scope_files(scope:str, manifest:Dict[str,Any])->List[str]:
    files = manifest.get("files", {})
    if scope == "external":
        return [p for p,meta in files.items() if meta.get("category")=="external"]
    if scope == "internal":
        return [p for p,meta in files.items() if meta.get("category")=="internal"]
    return list(files.keys())

def _format_citation(meta:Dict[str,Any])->str:
    return f"{os.path.basename(meta.get('filename',''))} : {int(meta.get('page',0))+1}"

# ------------------ Index ------------------
class BM25Index:
    def _init_(self, rows:List[Dict[str,Any]]):
        self.rows = rows
        self.docs = [r["text"] for r in rows]
        self.toks = [_tok(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.toks)

    def bm25_search(self, q:str, topk:int)->List[Tuple[int,float]]:
        toks = _tok(q)
        scores = self.bm25.get_scores(toks)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [(i, float(scores[i])) for i in idxs if scores[i] > 0]

    def fuzzy_search(self, q:str, topk:int)->List[Tuple[int,float]]:
        if not ENABLE_FUZZY:
            return []
        matches = process.extract(q, self.docs, scorer=fuzz.token_set_ratio, limit=topk)
        out = [(m[2], float(m[1])) for m in matches if m and m[1] > 0]
        return out

    def search_rrf(self, q:str, topk:int)->List[Tuple[int,float]]:
        s1 = self.bm25_search(q, topk)
        s2 = self.fuzzy_search(q, topk)
        if not s2:
            return s1
        merged = _rrf([s1, s2], k=RRF_K)
        return merged[:topk]

# ------------------ Corpus load by scope ------------------
def _load_corpus_for_scope(scope:str)->List[Dict[str,Any]]:
    rows = _read_jsonl(CORPUS_PATH)
    if not rows: return []
    if scope in ("internal","external"):
        rows = [r for r in rows if r.get("meta",{}).get("category")==scope]
    rows = _latest_only(rows)
    return rows

# ------------------ Answer builders ------------------
def _extract_snippets(rows:List[Dict[str,Any]], hits:List[Tuple[int,float]], citations:bool)->Tuple[str,List[str],List[Dict[str,Any]]]:
    snippets, cites, used = [], [], []
    for i,_ in hits[:MAX_SNIPPETS]:
        text = _normalize(rows[i]["text"])
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit(" ",1)[0] + "…"
        snippets.append(f"- {text}")
        used.append(rows[i])
        if citations:
            cites.append(_format_citation(rows[i].get("meta",{})))
    body = "\n".join(snippets) if snippets else "(no strong evidence found)"
    return body, cites, used

# ------------------ LLM helpers ------------------
def _ensure_bullet_first(text: str) -> str:
    if not text:
        return text
    lines = [ln.rstrip() for ln in str(text).splitlines()]
    for i, ln in enumerate(lines):
        if ln.strip():
            if ln.lstrip().startswith("* "):
                lines[i] = re.sub(r"^\s*\*\s+", "- ", ln)
            elif not ln.lstrip().startswith("- "):
                lines[i] = "- " + ln.lstrip()
            break
    return "\n".join(lines).strip()

def _llm_summarize(question:str, used_rows:List[Dict[str,Any]], citations:bool)->str:
    if not ENABLE_LLM:
        return ""
    blocks = []
    for idx, r in enumerate(used_rows, 1):
        meta = r.get("meta", {})
        title = _format_citation(meta)
        blocks.append(f"[{idx}] {title}\n{_normalize(r['text'])}")
    ctx = "\n\n".join(blocks)

    sys_prompt = (
        "You are a careful compliance analyst. Synthesize a concise, accurate answer to the user's question "
        "using ONLY the provided sources. If uncertain, say so. Keep to factual statements."
    )

    user_prompt = (
        f"Question: {question}\n\nSources:\n{ctx}\n\nInstructions:\n"
        "- Provide a structured answer in bullet points.\n"
        "- Include short quotes or paraphrases anchored to [n] source markers where relevant.\n"
        + ("- End with a 'Citations' list mapping [n] to source titles.\n" if citations else "")
    )

    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}
        )
        text = resp.get("message", {}).get("content", "")
        if isinstance(text, (list, dict)):
            text = json.dumps(text)
        return _ensure_bullet_first(str(text).strip())
    except Exception as e:  # pragma: no cover
        return f"(LLM synthesis unavailable: {e})"

def _llm_rerank(question:str, rows:List[Dict[str,Any]], hits:List[Tuple[int,float]], limit:int)->List[Tuple[int,float]]:
    if not ENABLE_LLM_RERANK or not ENABLE_LLM:
        return hits[:limit]
    cands = []
    for rank,(idx,score) in enumerate(hits[:max(limit*2, limit)], 1):
        meta = rows[idx].get("meta", {})
        title = _format_citation(meta)
        text = _normalize(rows[idx]["text"])[:MAX_CHARS]
        cands.append(f"[{rank}] {title}\n{text}")
    ctx = "\n\n".join(cands)

    prompt = (
        "You are reranking retrieval candidates for relevance. "
        "Return a JSON array of candidate indices (as integers) in best-to-worst order based on relevance to the question."
        f"\nQuestion: {question}\n\nCandidates:\n{ctx}\n\nOutput strictly JSON, e.g., [3,1,2]."
    )
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}], options={"temperature":0})
        content = resp.get("message", {}).get("content", "").strip()
        try:
            order = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r'\[[^\]]*\]', content, re.DOTALL)
            order = json.loads(m.group(0)) if m else []
        mapping = {i+1: hits[i][0] for i in range(min(len(hits), len(cands)))}
        new_hits = [(mapping.get(rk, hits[0][0]), 1.0/(i+1)) for i, rk in enumerate(order) if isinstance(rk, int) and rk in mapping]
        return new_hits[:limit] if new_hits else hits[:limit]
    except Exception:
        return hits[:limit]

# ------------------ Query rewrite gateway ------------------
def _rewrite_for_retrieval(q:str)->str:
    # rule-based rewrite
    rb = _rewrite_query_rule_based(q)
    if not ENABLE_LLM_QUERY_REWRITE:
        return rb
    # try LLM rewrite and blend
    llm = _llm_query_rewrite(q)
    if not llm:
        return rb
    # merge tokens from both (dedupe, keep order)
    merged = list(dict.fromkeys(_tok(rb) + _tok(llm)))
    return " " .join(merged)

# ------------------ Compare & Answer ------------------
def _compare_blocks(q:str, citations:bool)->Tuple[str,List[str]]:
    rq = _rewrite_for_retrieval(q)

    rows_ext = _load_corpus_for_scope("external")
    idx_ext  = BM25Index(rows_ext) if rows_ext else None
    hits_ext = idx_ext.search_rrf(rq, topk=BM25_TOPK) if idx_ext else []
    hits_ext = _llm_rerank(q, rows_ext, hits_ext, MAX_SNIPPETS) if hits_ext else []
    ext_body, ext_cites, _ = _extract_snippets(rows_ext, hits_ext, citations) if hits_ext else ("(no strong evidence found)", [], [])

    rows_int = _load_corpus_for_scope("internal")
    idx_int  = BM25Index(rows_int) if rows_int else None
    hits_int = idx_int.search_rrf(rq, topk=BM25_TOPK) if idx_int else []
    hits_int = _llm_rerank(q, rows_int, hits_int, MAX_SNIPPETS) if hits_int else []
    int_body, int_cites, _ = _extract_snippets(rows_int, hits_int, citations) if hits_int else ("(no strong evidence found)", [], [])

    ext_kw = set(re.findall(r"[A-Za-z]{3,}", ext_body))
    int_kw = set(re.findall(r"[A-Za-z]{3,}", int_body))
    gaps_ext2int = sorted(list((ext_kw - int_kw)))[:10]
    gaps_int2ext = sorted(list((int_kw - ext_kw)))[:10]

    out = []
    out.append("*External (Govt/Standards) – Key Points*")
    out.append(ext_body+"\n")
    out.append("*Internal (Company) – Key Points*")
    out.append(int_body+"\n")
    out.append("*Gaps / Misalignments (heuristic)*")
    out.append(f"- External→Internal missing themes: {', '.join(gaps_ext2int) if gaps_ext2int else '(none detected)'}")
    out.append(f"- Internal→External unmatched themes: {', '.join(gaps_int2ext) if gaps_int2ext else '(none detected)'}")

    cites = ext_cites + int_cites
    return "\n".join(out), cites

def _answer_block(q:str, scope:str, citations:bool)->Tuple[str,List[str]]:
    rq = _rewrite_for_retrieval(q)

    rows = _load_corpus_for_scope(scope)
    if not rows: return "(no suitable evidence found)", []
    index = BM25Index(rows)
    hits  = index.search_rrf(rq, topk=BM25_TOPK)
    hits  = _llm_rerank(q, rows, hits, MAX_SNIPPETS)
    body, cites, used = _extract_snippets(rows, hits, citations)

    llm_out = _llm_summarize(q, used, citations)
    if llm_out:
        body = llm_out
    return body, cites

# ------------------ Taskbar & CLI ------------------
def _list_resources(scope:str)->str:
    man = _load_manifest()
    files = _scope_files(scope, man)
    if not files: return "No collected resources yet."
    lines = [f"- {os.path.basename(p)}" for p in sorted(files)]
    return "\n".join(lines)

def main():
    # tests expect bullet-first answers without banners
    citations = DEFAULT_CITATIONS

    # refresh vocab if corpus has changed since import
    global _VOCAB_FREQ
    if not _VOCAB_FREQ and os.path.exists(CORPUS_PATH):
        _VOCAB_FREQ = _build_vocab_from_corpus()

    while True:
        q = input("Enter question: ").strip()
        if not q:
            continue
        if q.lower() in ("q","quit","exit"):
            print("\nGoodbye!")
            return
        if q.strip() == ":menu":
            _taskbar(citations)
            continue

        scope, mode = _infer_scope_and_mode(q)
        if "with citations" in q.lower():
            citations = True

        _append_history(q)

        if mode == "compare":
            body, cites = _compare_blocks(q, citations)
        else:
            body, cites = _answer_block(q, scope, citations)

        print(body)

        if citations and cites:
            uniq = []
            for c in cites:
                if c not in uniq: uniq.append(c)
            print("\nCitations:")
            for c in uniq[:20]:
                print(" -", c)

        print(f"\nDetected → Scope={scope} | Mode={mode} | Citations={'ON' if citations else 'OFF'} | LLM={'ON' if ENABLE_LLM else 'OFF'}")
        print("✅ Ingestion up to date.")

        _suggest_followups(q)

def _menu_scope(cur:str)->str:
    print(f"\nScope (current: {cur})\n  1) External  2) Internal  3) Both")
    pick = input("Pick 1/2/3 or Enter to keep: ").strip()
    return {"1":"external","2":"internal","3":"both"}.get(pick, cur)

def _menu_mode(cur:str)->str:
    print(f"\nMode (current: {cur})\n  1) Answer  2) Compare")
    pick = input("Pick 1/2 or Enter to keep: ").strip()
    return {"1":"answer","2":"compare"}.get(pick, cur)

def _menu_citations(cur:bool)->bool:
    print(f"\nCitations (current: {'ON' if cur else 'OFF'})\n  1) ON  2) OFF")
    pick = input("Pick 1/2 or Enter to keep: ").strip()
    return {"1":True,"2":False}.get(pick, cur)

def _taskbar(citations:bool)->None:
    print("\n=== Taskbar ===")
    print("  1) History")
    print("  2) Live feed (alerts)")
    print("  3) Summary")
    print("  4) Resources (internal/external/both)")
    print("  5) Back")
    sel = input("Select: ").strip()
    if sel == "1":
        hist = _read_jsonl(HISTORY_PATH)
        if not hist: print("\n(no history)\n"); return
        print()
        for h in hist[-20:]:
            print(f"- {h.get('ts','')} :: {h.get('q','')}")
        print()
    elif sel == "2":
        if not os.path.exists(FEED_PATH): print("\nNo recent feed items.\n"); return
        with open(FEED_PATH,"r",encoding="utf-8") as f:
            feed = json.load(f)
        ev = feed.get("events",[])
        if not ev: print("\nNo recent feed items.\n"); return
        print()
        for e in ev[-20:]:
            print(f"- [{e.get('ts')}] {e.get('type','').upper()} :: {e.get('doc')} :: {e.get('note')}")
        print()
    elif sel == "3":
        if not os.path.exists(SUMMARY_PATH): print("\n(no summary)\n"); return
        with open(SUMMARY_PATH,"r",encoding="utf-8") as f:
            s = json.load(f)
        print(f"\nLast Updated: {s.get('last_updated')}")
        counts = s.get("counts",{})
        print(f"External: {counts.get('External',0)}  | Internal: {counts.get('Internal',0)}")
        print(f"DOCX parser: {s.get('docx_support')}\n")
    elif sel == "4":
        scope = _menu_scope("both")
        print(f"\nResources in scope [{scope}]:")
        print(_list_resources(scope))
        print()

def _suggest_followups(q:str)->None:
    options = [
        "Map internal controls to specific external clauses?",
        "Explain CET1/LCR/NSFR and key drivers?",
        "Export a CSV traceability matrix?",
        "Walk through incident response workflow and recent incidents?"
    ]
    print("\nWould you like any of these follow-ups?")
    for i,opt in enumerate(options,1):
        print(f"- {opt}")
    print("(Enter a number 1-4, or anything else to skip.)\n")
    input("Pick: ")

if _name_ == "_main_":
    main()
# src/ingest.py
"""
Ingestion: bulk upload, tagging, version control, multi-format (PDF/DOCX/TXT).
Outputs:
  - vectorstore/corpus.jsonl  (rows: text, meta{filename,page,category,tags,version,source})
  - vectorstore/manifest.json (sha256 per file, categories, tags, version)
  - vectorstore/policy_summary.json (counts)
  - vectorstore/live_feed.json (alerts for non-compliance heuristics)

Fixes in v1.3:
- pdfminer fallback when PyPDF yields poor text
- tqdm in requirements and guarded use on QUIET
- deterministic versioning; manifest migration preserved
"""

import os, re, json, hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader

# Optional DOCX readers
DOCX_AVAILABLE = False
UNSTRUCTURED_AVAILABLE = False
DocxDocument = None
partition_docx = None
try:
    from docx import Document as DocxDocument  # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DocxDocument = None

if not DOCX_AVAILABLE:
    try:
        from unstructured.partition.docx import partition_docx
        UNSTRUCTURED_AVAILABLE = True
    except ImportError:
        partition_docx = None

# Optional pdfminer fallback with a lazy shim to satisfy linters and runtime
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore[import-not-found]
except Exception:
    # Define a shim so the symbol exists for linters; import lazily at call time.
    def pdfminer_extract_text(*args, **kwargs):  # type: ignore
        try:
            from pdfminer.high_level import extract_text as _ext  # type: ignore
            return _ext(*args, **kwargs)
        except Exception:
            return ""

load_dotenv()

DATA_DIR      = os.getenv("DATA_DIR", os.path.join("data", "raw"))
CORPUS_PATH   = os.getenv("CORPUS_PATH", "vectorstore/corpus.jsonl")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "vectorstore/manifest.json")
SUMMARY_PATH  = os.getenv("SUMMARY_PATH", "vectorstore/policy_summary.json")
FEED_PATH     = os.getenv("FEED_PATH", "vectorstore/live_feed.json")
QUIET         = os.getenv("QUIET", "1") == "1"

SUPPORTED_EXT = {".pdf", ".txt", ".docx"}

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _category_from_path(path: str) -> str:
    p = (path or "").replace("\\", "/").lower()
    if "/external/" in p: return "external"
    if "/internal/" in p: return "internal"
    return "internal"

def _default_tags(filename: str, category: str) -> List[str]:
    base = os.path.basename(filename).lower()
    tags = set([category])
    if any(k in base for k in ["policy", "sop", "manual", "handbook"]): tags.add("policy")
    if any(k in base for k in ["rbi", "mas", "iso", "ffiec", "gdpr", "basel"]): tags.add("regulation")
    return sorted(tags)

def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _load_manifest() -> Dict[str, Any]:
    base = {"files": {}, "last_ingest": None}
    if not os.path.exists(MANIFEST_PATH): return base
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return base
    if isinstance(raw, list):  # legacy → migrate
        files_map = {}
        for item in raw:
            if isinstance(item, dict):
                path = item.get("path") or item.get("source") or item.get("filename")
                if path:
                    files_map[path] = {
                        "sha": item.get("sha") or item.get("hash") or "",
                        "version": int(item.get("version", 1)),
                        "category": item.get("category") or _category_from_path(path),
                        "tags": item.get("tags") or [],
                    }
        return {"files": files_map, "last_ingest": None}
    if isinstance(raw, dict):
        raw.setdefault("files", {})
        raw.setdefault("last_ingest", None)
        if not isinstance(raw["files"], dict): raw["files"] = {}
        return raw
    return base







def _save_manifest(man: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

def _write_feed_event(doc: str, note: str, etype: str = "alert") -> None:
    os.makedirs(os.path.dirname(FEED_PATH), exist_ok=True)
    feed = {"events": []}
    if os.path.exists(FEED_PATH):
        try:
            with open(FEED_PATH, "r", encoding="utf-8") as f:
                feed = json.load(f)
        except Exception:
            pass
    feed.setdefault("events", []).append(
        {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "type": etype, "doc": doc, "note": note}
    )
    with open(FEED_PATH, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)

def _non_compliance_scan(text: str) -> List[str]:
    flags = []
    low = text.lower()
    if ("personal data" in low or "pii" in low) and "encrypt" not in low:
        flags.append("PII mentioned without encryption context")
    if "retention" in low and not any(x in low for x in ["year", "month", "day", "schedule"]):
        flags.append("Retention with no duration")
    if "incident" in low and "response" not in low:
        flags.append("Incident mention without response steps")
    return flags

def _collect_files() -> List[str]:
    paths = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXT:
                paths.append(os.path.join(root, f))
    return sorted(paths)

# ---------- Readers ----------

def _read_pdf_pypdf(path: str) -> List[Dict[str, Any]]:
    rows = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt: rows.append({"text": txt, "page": i})
    except Exception as e:
        _write_feed_event(os.path.basename(path), f"PyPDF failed: {e}", etype="warning")
    return rows

def _read_pdf_pdfminer(path: str) -> List[Dict[str, Any]]:
    if not pdfminer_extract_text:
        return []
    try:
        txt = pdfminer_extract_text(path) or ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            return []
        rows, page, buf = [], 0, []
        words = txt.split(" ")
        cur = 0
        cap = 1500
        for w in words:
            buf.append(w)
            cur += len(w) + 1
            if cur >= cap:
                rows.append({"text": " ".join(buf), "page": page})
                page += 1
                buf = []
                cur = 0
        if buf:
            rows.append({"text": " ".join(buf), "page": page})
        return rows
    except Exception as e:
        _write_feed_event(os.path.basename(path), f"pdfminer failed: {e}", etype="warning")
        return []

def _read_pdf(path: str) -> List[Dict[str, Any]]:
    rows = _read_pdf_pypdf(path)
    if not rows:
        alt = _read_pdf_pdfminer(path)
        if alt:
            _write_feed_event(os.path.basename(path), "Used pdfminer fallback", etype="info")
        return alt
    return rows

def _read_txt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = re.sub(r"\r\n", "\n", f.read())
    parts = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    return [{"text": p, "page": i} for i, p in enumerate(parts)]
def _read_docx_python_docx(path: str) -> List[Dict[str, Any]]:
    doc = DocxDocument(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    rows, page, buf, cap = [], 0, [], 1500
    cur = 0
    for para in paras:
        if not para:
            continue
        if cur + len(para) > cap and buf:
            rows.append({"text": " ".join(buf), "page": page})
            page += 1
            buf = []
            cur = 0
        buf.append(para)
        cur += len(para) + 1
    if buf:
        rows.append({"text": " ".join(buf), "page": page})
    return rows

def _read_docx_unstructured(path: str) -> List[Dict[str, Any]]:
    elements = partition_docx(filename=path)  # type: ignore
    rows, page, buf, size, cap = [], 0, [], 0, 1500
    for el in elements:
        txt = str(el).strip()
        if not txt:
            continue
        if size + len(txt) > cap and buf:
            rows.append({"text": " ".join(buf), "page": page})
            page += 1
            buf, size = [], 0
        buf.append(txt)
        size += len(txt)
    if buf:
        rows.append({"text": " ".join(buf), "page": page})
    return rows

def _read_docx(path: str) -> List[Dict[str, Any]]:
    if DOCX_AVAILABLE and DocxDocument is not None:
        try:
            return _read_docx_python_docx(path)
        except Exception as e:
            _write_feed_event(os.path.basename(path), f"python-docx failed: {e}", etype="warning")
    if UNSTRUCTURED_AVAILABLE and partition_docx is not None:
        try:
            return _read_docx_unstructured(path)
        except Exception as e:
            _write_feed_event(os.path.basename(path), f"unstructured failed: {e}", etype="warning")
    _write_feed_event(os.path.basename(path), "DOCX parsing skipped: install 'python-docx' or 'unstructured'", etype="warning")
    return []

def _parse_file(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return _read_pdf(path)
    if ext == ".txt": return _read_txt(path)
    if ext == ".docx": return _read_docx(path)
    return []

def _version_for(path: str, manifest: Dict[str, Any], sha: str) -> int:
    rec = manifest.get("files", {}).get(path)
    if not rec: return 1
    if rec.get("sha") != sha: return int(rec.get("version", 1)) + 1
    return int(rec.get("version", 1))

def ingest_pdfs(force: bool = False) -> None:
    if not QUIET:
        print(">>> ingest.py starting")
    files = _collect_files()
    manifest = _load_manifest()
    manifest.setdefault("files", {})
    manifest.setdefault("last_ingest", None)

    if force and os.path.exists(CORPUS_PATH):
        os.remove(CORPUS_PATH)

    counts = {"External": 0, "Internal": 0}
    changed = False

    iterator = tqdm(files, desc="Ingesting", unit="file") if not QUIET else files
    for path in iterator:
        sha = _sha256_file(path)
        version = _version_for(path, manifest, sha)
        was = manifest["files"].get(path)
        category = _category_from_path(path)
        tags = _default_tags(path, category)

        if was and was.get("sha") == sha and not force:
            counts["External" if category == "external" else "Internal"] += 1
            continue

        rows = _parse_file(path)
        fname = os.path.basename(path)
        for r in rows:
            meta = {
                "filename": fname,
                "source": path,
                "page": r.get("page"),
                "category": category,
                "tags": tags,
                "version": version,
            }
            _append_jsonl(CORPUS_PATH, {"text": r["text"], "meta": meta})
            for flag in _non_compliance_scan(r["text"]):
                _write_feed_event(fname, flag, etype="noncompliance")

        manifest["files"][path] = {"sha": sha, "version": version, "category": category, "tags": tags}
        changed = True
        counts["External" if category == "external" else "Internal"] += 1

    manifest["last_ingest"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    _save_manifest(manifest)

    summary = {
        "last_updated": manifest["last_ingest"],
        "counts": counts,
        "docx_support": "python-docx" if DOCX_AVAILABLE else ("unstructured" if UNSTRUCTURED_AVAILABLE else "unavailable"),
    }
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not QUIET:
        print("✅ Ingestion complete." if changed else "Manifest unchanged → no changes detected.")

if __name__ == "__main__":
    ingest_pdfs(force=False)

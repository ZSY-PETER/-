# -*- coding: utf-8 -*-
"""Bridge RAG v2.0: precise clause lookup first, vector search second.

Reads the ORIGINAL ChromaDB in-place (read/query only). Does not create a
collection, rebuild embeddings, or mutate stored data.

Compatible with previous C++ call:
    python rag_query.py input.json output.json
    python rag_query.py --check

Input JSON legacy: {"query":"...", "top_k":5}
Optional: standard_id, clause_id, component_type, damage_type.

Reliability notes:
- Clause location is algorithmically extracted, not manually checked against
  an official original. Results must be verified before engineering decisions.
- Synthetic millimetre measurements must never be reported as actual measurements.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from utf8_process import set_utf8_stdio
from collections import Counter
from pathlib import Path

OLLAMA_URL = os.environ.get("BRIDGE_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.environ.get("BRIDGE_EMBEDDING_MODEL", "bge-m3:latest")
COLLECTION_NAME = "standards_ollama"
DEFAULT_TOP_K = 5
CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATES = 20
MAX_RESULTS_PER_SOURCE = 2
DEFAULT_DB_ROOT = Path.home() / "Desktop" / "新建文件夹 (2)"
MAX_CONTEXT_CHARS = 14000


def locate_db(root):
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"数据库根目录不存在：{root}")
    choices = []
    for p in (root, root / "chroma_db"):
        if (p / "chroma.sqlite3").is_file():
            choices.append(p)
    if not choices:
        choices = sorted(set(p.parent for p in root.rglob("chroma.sqlite3")))
    if not choices:
        raise FileNotFoundError(f"{root} 下未发现 chroma.sqlite3。不会自动创建空库。")
    if len(choices) > 1:
        raise RuntimeError("发现多个 ChromaDB，请使用 --db 选择：\n" + "\n".join(map(str, choices)))
    return choices[0]


def inspect_db(folder):
    db = Path(folder) / "chroma.sqlite3"
    uri = db.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        collections = [row[0] for row in conn.execute("SELECT name FROM collections")]
    print(f"[CHECK] ChromaDB: {folder}")
    print(f"[CHECK] 数据库文件: {db} ({db.stat().st_size} bytes)")
    print(f"[CHECK] Collections: {', '.join(collections)}")
    print("[CHECK] 未修改向量库；--check 不验证模型、条款质量。")
    return COLLECTION_NAME in collections


def normalize_standard_id(value):
    if not value:
        return ""
    s = str(value).upper().strip()
    s = s.replace("—", "-").replace("－", "-").replace("–", "-").replace("_", " ")
    # Older document filenames store the slash as a hyphen.
    s = re.sub(r"\b(JTG|GB|CJJ|JGJ|JT|TB)\s*[-/]\s*T\s*", r"\1/T ", s)
    s = re.sub(r"\b(GBT|JTG T|CJJT|JGJT|JTT|TBT)\s*(?=[A-Z]?\d)",
               lambda m: {"GBT": "GB/T", "JTG T": "JTG/T", "CJJT": "CJJ/T",
                          "JGJT": "JGJ/T", "JTT": "JT/T", "TBT": "TB/T"}[m.group(1)] + " ", s)
    # OCR occasionally prints JTG/TH21 without a space.
    s = re.sub(r"\b(JTG/T|GB/T|CJJ/T|JGJ/T|JT/T|TB/T)\s*([A-Z]?\d)", r"\1 \2", s)
    s = re.sub(r"\b(JTG|GB|CJJ|JGJ|JTJ|JT|TB)\s*([A-Z]?\d)", r"\1 \2", s)
    s = re.sub(r"\s*-\s*(\d{4})\b", r"-\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


STANDARD_PATTERN = re.compile(
    r"(?<![A-Z0-9])((?:JTG/T|JTG|JT/T|JTJ|GB/T|GB|CJJ/T|CJJ|JGJ/T|JGJ|TB/T|TB)"
    r"\s*[A-Z]?\s*\d+(?:\.\d+)*(?:\s*-\s*\d{4})?)",
    re.IGNORECASE,
)


def standard_id_in_text(value):
    s = normalize_standard_id(value)
    m = STANDARD_PATTERN.search(s)
    return normalize_standard_id(m.group(1)) if m else ""


def source_from_metadata(metadata):
    m = metadata or {}
    filename = str(m.get("file_name") or m.get("source_file") or "未知来源")
    identifier = standard_id_in_text(m.get("standard_id") or "")
    if not identifier:
        identifier = standard_id_in_text(filename)
    return filename, identifier, str(m.get("standard_name") or "")


def clause_from_query(query, explicit=None):
    if explicit is not None and str(explicit).strip():
        candidate = str(explicit).strip()
        if not re.fullmatch(r"\d+(?:\.\d+){1,4}", candidate):
            raise ValueError(f"clause_id 格式不正确: {candidate}")
        return candidate
    q = str(query)
    m = re.search(r"第\s*(\d+(?:\.\d+){1,4})\s*条", q)
    if not m:
        m = re.search(r"(?<!\w)(?:表|条款|条文)\s*(\d+(?:\.\d+){1,4})(?![\d.])", q)
    return m.group(1) if m else ""


def find_clause_headings(document, clause):
    if not clause:
        return []
    escaped = r"\s*\.\s*".join(re.escape(x) for x in clause.split("."))
    # Must start a line: avoid table references such as '见表9.1.2'.
    rgx = re.compile(r"(?m)^[ \t]*(?:第[ \t]*)?" + escaped +
                     r"(?![.\d])(?:[ \t]*条)?(?=[ \t]*[\u4e00-\u9fff]|[ \t]*[:：])")
    return list(rgx.finditer(document or ""))


NEXT_HEADING = re.compile(r"(?m)^[ \t]*(\d+(?:\.\d+){1,4})(?![.\d])(?=[ \t]*[\u4e00-\u9fff]|[ \t]*[:：])")


def extract_clause(document, clause):
    """Return clause text, true if another section heading ends it."""
    starts = find_clause_headings(document, clause)
    if not starts:
        return "", False
    start = starts[0].start()
    end = len(document)
    finished = False
    for m in NEXT_HEADING.finditer(document, starts[0].end()):
        if m.group(1) == clause:
            continue
        end, finished = m.start(), True
        break
    return document[start:end].strip(), finished


def overlap_merge(left, right):
    """Merge overlapping indexed chunks while preserving the original text."""
    if not left:
        return right
    if not right or right in left:
        return left
    possible = min(len(left), len(right), 1600)
    for n in range(possible, 29, -1):
        if left.endswith(right[:n]):
            return left + right[n:]
    return left + "\n\n[相邻文本块，未能确认重叠边界]\n" + right


def context_cut(text):
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return text[:MAX_CONTEXT_CHARS] + "\n[内容过长已截断，不能据此断言条文完整]"


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def result_location(meta, source, sid, name, *, clause="", verified=False):
    idx = safe_int((meta or {}).get("chunk_index", (meta or {}).get("index", -1)), -1)
    return {
        "file_name": source, "standard_name": name, "standard_id": sid,
        "chapter": str((meta or {}).get("chapter") or ""),
        "section": clause or str((meta or {}).get("section") or ""),
        "clause_hint_unverified": clause if not verified else "",
        "chunk_index": idx,
        "description": f"源文件第 {idx + 1} 个文本块附近（非精确页码）" if idx >= 0 else "来源位置未核实",
        "citation_verified": bool(verified),
    }


class RAGKnowledgeBaseOllama:
    def __init__(self, db_path=None):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("请先安装：python -m pip install chromadb requests") from exc
        self.db_path = locate_db(db_path or os.environ.get("BRIDGE_CHROMA_DB") or DEFAULT_DB_ROOT)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)  # never creates
        self.count = self.collection.count()
        if not self.count:
            raise RuntimeError("standards_ollama Collection 为空")
        self.metric = (self.collection.metadata or {}).get("hnsw:space", "未知")
        self._source_cache = None
        print(f"[RAG] 数据库：{self.db_path}")
        print(f"[RAG] Collection：{COLLECTION_NAME}，文档块：{self.count}，距离：{self.metric}")

    def all_source_files(self):
        if self._source_cache is not None:
            return self._source_cache
        names = {}
        # Metadata scan: 951 records => ~2 calls; no document embeddings involved.
        for offset in range(0, self.count, 500):
            chunk = self.collection.get(limit=min(500, self.count - offset), offset=offset,
                                        include=["metadatas"])
            for meta in chunk.get("metadatas") or []:
                if (meta or {}).get("doc_type", "standard") != "standard":
                    continue
                name, sid, _ = source_from_metadata(meta)
                if name != "未知来源":
                    names[name] = sid
        self._source_cache = names
        return names

    def identify_sources(self, standard_id):
        if not standard_id:
            return list(self.all_source_files())
        key = normalize_standard_id(standard_id)
        return [name for name, sid in self.all_source_files().items() if sid == key]

    def source_chunks(self, filename):
        data = self.collection.get(where={"file_name": filename},
                                   include=["documents", "metadatas"])
        rows = []
        for i, row_id in enumerate(data.get("ids") or []):
            meta = (data.get("metadatas") or [])[i] or {}
            if meta.get("doc_type", "standard") != "standard":
                continue
            doc = (data.get("documents") or [])[i] or ""
            idx = safe_int(meta.get("chunk_index", meta.get("index", -1)), -1)
            rows.append({"id": row_id, "metadata": meta, "document": doc, "index": idx})
        return sorted(rows, key=lambda r: r["index"])

    def exact_clause_search(self, clause, standard_id=""):
        """Find exact numbered heading, extract body up to the next heading.

        This is a *textual candidate*, not an official verified standard citation.
        """
        sources = self.identify_sources(standard_id)
        if standard_id and not sources:
            print(f"[RAG] 未找到指定标准来源：{standard_id}，不跨标准退化检索。")
            return []
        matched = []
        for filename in sources:
            rows = self.source_chunks(filename)
            by_index = {r["index"]: r for r in rows if r["index"] >= 0}
            for item in rows:
                if not find_clause_headings(item["document"], clause):
                    continue
                body, ended = extract_clause(item["document"], clause)
                used = [item["index"]]
                if not ended:
                    # Only extend while the next heading has not been encountered.
                    for next_i in range(item["index"] + 1, item["index"] + 4):
                        nxt = by_index.get(next_i)
                        if nxt is None:
                            break
                        joined = overlap_merge(item["document"], nxt["document"])
                        body2, finished = extract_clause(joined, clause)
                        if body2:
                            body, ended = body2, finished
                            used.append(next_i)
                        if ended:
                            break
                source, sid, standard_name = source_from_metadata(item["metadata"])
                evidence = {
                    "raw_rank": 0, "id": item["id"],
                    "document": context_cut(body), "metadata": item["metadata"],
                    "location": result_location(item["metadata"], source, sid, standard_name,
                                                clause=clause, verified=False),
                    "distance": None, "score": None,
                    "file_name": source, "standard_name": standard_name,
                    "standard_id": sid, "category": item["metadata"].get("category", ""),
                    "retrieval_method": "exact_clause_heading",
                    "clause_id": clause,
                    "source_chunk_indices": sorted(set(used)),
                    "clause_text_may_be_incomplete": (not ended or len(body) > MAX_CONTEXT_CHARS),
                    "citation_verified": False,
                    "applicability_status": "requires_engineering_review",
                }
                matched.append(evidence)
        # Chunk 56 and 57 both have heading; prefer the *most complete* extracted text.
        best = {}
        for item in matched:
            key = (item["file_name"], item["clause_id"])
            if key not in best or len(item["document"]) > len(best[key]["document"]):
                best[key] = item
        ordered = sorted(best.values(), key=lambda r: (-len(r["document"]), r["file_name"]))
        for i, item in enumerate(ordered, 1):
            item["raw_rank"] = i
            item["diversified_rank"] = i
        return ordered

    def get_embedding(self, query):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("缺少 requests，请安装 pip install requests") from exc
        if not query.strip():
            raise ValueError("查询字符串为空")
        response = requests.post(OLLAMA_URL + "/api/embed",
                                 json={"model": EMBEDDING_MODEL, "input": query[:3000]},
                                 timeout=120)
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        if not embeddings or not isinstance(embeddings[0], list):
            raise RuntimeError("Ollama /api/embed 未返回合法向量")
        return embeddings[0]

    def semantic_search(self, query, top_k, standard_id=""):
        sources = self.identify_sources(standard_id) if standard_id else []
        if standard_id and not sources:
            print(f"[RAG] 目标标准未在库内找到：{standard_id}")
            return []
        n_candidates = min(self.count, max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER))
        args = {"query_embeddings": [self.get_embedding(query)],
                "n_results": n_candidates,
                "include": ["documents", "metadatas", "distances"]}
        if standard_id:
            args["where"] = {"file_name": sources[0]} if len(sources) == 1 else {"file_name": {"$in": sources}}
        else:
            args["where"] = {"doc_type": "standard"}
        try:
            data = self.collection.query(**args)
        except Exception as exc:
            raise RuntimeError(f"向量检索失败：{exc}") from exc
        ids = (data.get("ids") or [[]])[0]
        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]
        pool = []
        for i, row_id in enumerate(ids):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            if meta.get("doc_type", "standard") != "standard":
                continue
            doc = docs[i] if i < len(docs) else ""
            dist = distances[i] if i < len(distances) else None
            src, sid, name = source_from_metadata(meta)
            pool.append({
                "raw_rank": i + 1, "id": row_id, "document": context_cut(doc),
                "metadata": meta,
                "location": result_location(meta, src, sid, name),
                "distance": dist, "score": 1 - dist if dist is not None and self.metric == "cosine" else None,
                "file_name": src, "standard_name": name, "standard_id": sid,
                "category": meta.get("category", ""), "retrieval_method": "semantic_candidate",
                "clause_id": "", "citation_verified": False,
                "applicability_status": "requires_engineering_review",
            })
        used, counts, selected = set(), Counter(), []
        for cap in (MAX_RESULTS_PER_SOURCE, float("inf")):
            for item in pool:
                if len(selected) >= top_k:
                    break
                if item["id"] in used or counts[item["file_name"]] >= cap:
                    continue
                used.add(item["id"])
                counts[item["file_name"]] += 1
                selected.append(item)
            if len(selected) >= top_k:
                break
        for j, item in enumerate(selected, 1):
            item["diversified_rank"] = j
        return selected

    def search(self, query, top_k=DEFAULT_TOP_K, *, standard_id="", clause_id=""):
        if clause_id:
            exact = self.exact_clause_search(clause_id, standard_id)
            if exact:
                print(f"[RAG] 精确条款定位：{standard_id or '任意标准'} 第{clause_id}条，"
                      f"命中来源 {len(exact)} 个；优先返回条款正文，不补充无关向量块。")
                return exact[:top_k], "exact_clause_candidate"
            print(f"[RAG WARN] 未定位到第{clause_id}条标题。向量结果只能作为候选，不能视为已定位条款。")
        items = self.semantic_search(query, top_k, standard_id)
        print(f"[RAG] 语义检索返回 {len(items)} 条候选")
        for item in items:
            score = f"{item['score']:.4f}" if item["score"] is not None else "N/A"
            print(f"  #{item['diversified_rank']} {item['file_name']} | {item['standard_id']} | score={score}")
        return items, "semantic_fallback" if clause_id else "semantic"

    def get_prompt_context(self, query, n_results=DEFAULT_TOP_K, *, standard_id="", clause_id=""):
        results, mode = self.search(query, n_results, standard_id=standard_id, clause_id=clause_id)
        if not results:
            return "未找到适用规范的可用检索证据，禁止补写条文或合格性结论。", [], mode
        blocks = ["【桥梁规范候选证据】", "提示：以下来源/条款为程序定位结果，未逐项核对正式标准原件；"
                  "应确认桥型、构件、裂缝类型、规范版本及表格排版后才能作工程判定。"]
        for item in results:
            loc = item["location"]
            blocks.append(f"[{item['diversified_rank']}] 来源：{item['file_name']}\n"
                          f"标准编号：{item['standard_id'] or '未核实'}\n"
                          f"条款：{item['clause_id'] or '尚未定位'}\n"
                          f"定位：第 {loc['chunk_index'] + 1} 个文本块（不是正式页码）\n"
                          f"证据方式：{item['retrieval_method']}；正式原文核验：否\n"
                          f"文本：\n{item['document']}\n")
        return context_cut("\n".join(blocks)), results, mode


def run(input_path, *, db_dir=None):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("输入 JSON 必须包含非空 query")
    top_k = max(1, min(30, safe_int(payload.get("top_k"), DEFAULT_TOP_K)))
    standard_id = standard_id_in_text(payload.get("standard_id") or query)
    clause_id = clause_from_query(query, payload.get("clause_id"))
    kb = RAGKnowledgeBaseOllama(db_path=db_dir)
    context, results, mode = kb.get_prompt_context(query, top_k,
                                                   standard_id=standard_id, clause_id=clause_id)
    return {
        "success": True,
        "query": query, "top_k": top_k,
        "candidate_strategy": {
            "retrieve_multiplier": CANDIDATE_MULTIPLIER,
            "min_candidates": MIN_CANDIDATES,
            "max_results_per_source": MAX_RESULTS_PER_SOURCE,
            "priority": "exact_clause_heading > constrained_semantic_search",
        },
        "retrieval_mode": mode, "requested_standard_id": standard_id,
        "requested_clause_id": clause_id,
        "engineering_inputs": {
            "component_type": str(payload.get("component_type") or ""),
            "damage_type": str(payload.get("damage_type") or ""),
        },
        "engineering_applicability_verified": False,
        "context": context, "results": results,
        "db_path": str(kb.db_path), "collection": COLLECTION_NAME,
        "measurement_warning": "模拟尺度仅用于软件演示；无可信物理标定不能输出实测毫米判定。",
        "citation_warning": "当前条款仍需与正式标准原件核对，不能将文本块当作准确页码。",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="桥梁 RAG v2：精确条款优先，兼容原 JSON 接口")
    parser.add_argument("input_json", nargs="?")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--db", default=None, help="数据库目录或包含数据库的上级目录")
    parser.add_argument("--check", action="store_true", help="只读检查数据库，不需要 Ollama")
    args = parser.parse_args(argv)
    if args.check:
        db = locate_db(args.db or os.environ.get("BRIDGE_CHROMA_DB") or DEFAULT_DB_ROOT)
        return 0 if inspect_db(db) else 1
    if not args.input_json or not args.output_json:
        parser.error("需要输入 JSON、输出 JSON，或使用 --check")
    out = Path(args.output_json)
    try:
        result = run(args.input_json, db_dir=args.db)
        exit_code = 0
    except Exception as exc:
        result = {"success": False, "error": str(exc), "context": "", "results": []}
        exit_code = 1
        print(f"[RAG ERROR] {exc}", file=sys.stderr)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RAG] JSON 已写出：{out}")
    except OSError as exc:
        print(f"[RAG ERROR] 保存 JSON 失败：{exc}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    set_utf8_stdio()
    sys.exit(main())

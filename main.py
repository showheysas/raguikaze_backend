# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from collections import defaultdict
import re

# 追加ライブラリ
from rank_bm25 import BM25Okapi
import nltk, re, numpy as np

# ───────────────────────────────────
# 0. 環境変数・ライブラリ準備
# ───────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# NLTK tokenizer 用リソース（初回のみ DL）
nltk.download("punkt", quiet=True)

# ───────────────────────────────────
# 1. Chroma と BM25 インデックス初期化
# ───────────────────────────────────
# 環境変数から Chroma 保存パスを取得（デフォルトは ./chroma_db）
chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")

chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma_client.get_or_create_collection(name="alctax-act")

def tokenize(text: str) -> list[str]:
    """
    極簡易トークナイザ:
    - 英数字は空白区切り
    - 日本語は 1 文字ずつ (BM25 は文字 N-gram でもそこそこ効く)
    """
    text = re.sub(r"\s+", " ", text.strip())
    tokens = []
    buf = ""
    for ch in text:
        if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff":
            # 日本語なら 1 文字単位で push
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            # 英数字はまとめて
            if ch.isspace():
                if buf:
                    tokens.append(buf)
                    buf = ""
            else:
                buf += ch
    if buf:
        tokens.append(buf)
    return tokens

# 全チャンクを取得して BM25 corpus 生成
all_data = collection.get(include=["documents"]) 
all_docs = all_data["documents"]
all_ids  = all_data["ids"]

print("Chroma documents:", len(all_docs))
print("Chroma DB path:", chroma_db_path)
print("Chroma collection name:", collection.name)
print("Chroma documents:", len(all_docs))

tokenized_corpus = [tokenize(t) for t in all_docs]
bm25 = BM25Okapi(tokenized_corpus)

# id ↔ index 対応表
id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
idx_to_id = {i: doc_id for i, doc_id in enumerate(all_ids)}

# ───────────────────────────────────
# 2. FastAPI 設定
# ───────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str
    kazemode: bool = False

class AskResponse(BaseModel):
    answer: str
    references: list

# ───────────────────────────────────
# 3. RRF 融合関数
# ───────────────────────────────────
def rrf_fuse(
    dense_ids: list[str],
    sparse_idxs: list[int],
    alpha: float = 0.5,
    k_dense: int = 30,
    k_sparse: int = 30,
) -> list[str]:
    """Reciprocal Rank Fusion（αで重み付け）"""
    ranks_dense  = {doc_id: r + 1 for r, doc_id in enumerate(dense_ids[:k_dense])}
    ranks_sparse = {idx_to_id[i]: r + 1 for r, i in enumerate(sparse_idxs[:k_sparse])}

    all_ids = set(ranks_dense) | set(ranks_sparse)
    fused = []
    for _id in all_ids:
        rd = ranks_dense.get(_id, 10**6)
        rs = ranks_sparse.get(_id, 10**6)
        score = alpha / rd + (1 - alpha) / rs
        fused.append((_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return [_id for _id, _ in fused]

# ───────────────────────────────────
# 4. /ask エンドポイント
# ───────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_law(req: AskRequest):
    # 4-1 密ベクトル検索
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    )
    query_vec = embed.data[0].embedding
    dense_res = collection.query(
        query_embeddings=[query_vec],
        n_results=30
    )
    dense_ids = dense_res["ids"][0]

    # 4-2 疎ベクトル（BM25）検索
    sparse_scores = bm25.get_scores(tokenize(req.query))
    top_sparse_idxs = np.argsort(sparse_scores)[::-1].tolist()

    # 4-3 RRF 融合
    fused_ids = rrf_fuse(dense_ids, top_sparse_idxs, alpha=0.5)

    # 4-4 メタデータ取得
    fused_docs = collection.get(ids=fused_ids)
    docs       = fused_docs["documents"]
    metas      = fused_docs["metadatas"]

    # prefLabel ごとに結合し上位 5 件採用
    grouped, id_lookup = defaultdict(list), {}
    for doc_id, doc, meta in zip(fused_ids, docs, metas):
        pref = meta.get("prefLabel", "")
        grouped[pref].append(doc)
        id_lookup.setdefault(pref, doc_id)

    top_contexts, references = [], []
    for i, (pref, chunk_list) in enumerate(grouped.items()):
        if i >= 5:
            break
        combined = "\n".join(chunk_list)
        top_contexts.append(combined)
        references.append({
            "id": id_lookup[pref],
            "text": combined[:200] + "...",
            "prefLabel": pref
        })

    context = "\n---\n".join(top_contexts)

    # 4-5 プロンプト
    if req.kazemode:
        style = (
            "あなたは藤井風本人です。"
            "以下の法令条文情報【のみ】に基づいて答えてください。"
            "例えや雑談を交えながら、口語体で、リラックスした雰囲気で答えてください。"
        )
    else:
        style = (
            "あなたは法律の専門家です。"
            "以下の法令条文情報【のみ】に基づいて、簡潔かつ正確に答えてください。"
            "一般常識や事前知識には頼らず、必ず条文内容を根拠にしてください。"
        )

    prompt = f"{style}\n\n# 質問:\n{req.query}\n\n# 法令条文情報:\n{context}"

    # 4-6 GPT 応答生成
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "references": references
    }

# ───────────────────────────────────
# 5. /log エンドポイント
# ───────────────────────────────────
from fastapi import Request
from fastapi.responses import JSONResponse
import csv
from datetime import datetime, timedelta, timezone
import pathlib

class LogRequest(BaseModel):
    query: str
    answer: str
    kazemode: bool
    timestamp: str  # ISO8601形式でFlutterから送信
    user_agent: str
    os: str

@app.post("/log")
async def log_entry(log: LogRequest):
    try:
        # 日本時間に変換
        dt_utc = datetime.fromisoformat(log.timestamp)
        JST = timezone(timedelta(hours=9))
        dt_jst = dt_utc.astimezone(JST)
        time_str = dt_jst.strftime("%Y-%m-%d %H:%M:%S")

        # CSVファイルに追記
        log_dir = pathlib.Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "raguikaze_logs.csv"

        write_header = not log_file.exists()
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "query", "answer", "kazemode", "user_agent", "os"])
            writer.writerow([
                time_str,
                log.query.replace("\n", " "),
                log.answer.replace("\n", " "),
                log.kazemode,
                log.user_agent,
                log.os,
            ])

        return JSONResponse(content={"message": "Log saved"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


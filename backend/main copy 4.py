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

# .env 読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI・Chroma初期化
client = OpenAI(api_key=openai_api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="alctax-act")

# FastAPI初期化
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエスト・レスポンス定義
class AskRequest(BaseModel):
    query: str
    kazemode: bool = False
    granularity: str = "all"  # "jou", "kou", "gou", "all"

class AskResponse(BaseModel):
    answer: str
    references: list

# 粒度に応じた除外関数
def is_filtered(pref_label: str, granularity: str) -> bool:
    if granularity == "jou":
        return bool(re.search(r"第[^条]+(項|号)[イロハニホヘトチリヌルヲ]*", pref_label))
    elif granularity == "kou":
        return bool(re.search(r"第[^条]+号[イロハニホヘトチリヌルヲ]*", pref_label))
    elif granularity == "gou":
        return bool(re.search(r"[イロハニホヘトチリヌルヲ]$", pref_label))
    return False  # all

# 再言語化（OpenAI APIで表現を拡張）
def rewrite_query(original_query: str) -> str:
    prompt = (
        "次の質問文を、酒税法の条文とよりマッチしやすいように、"
        "法令で使われる表現に書き換えてください。"
        "意味を変えず、専門的な言い回しを増やしてください。\n\n"
        f"質問: {original_query}\n\n"
        "書き換え:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@app.post("/ask", response_model=AskResponse)
async def ask_law(req: AskRequest):
    # 再言語化
    rewritten_query = rewrite_query(req.query)

    # ベクトル埋め込み（次元1536で登録されたChromaと一致）
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=rewritten_query
    )
    query_vec = embed.data[0].embedding

    # ベクトル検索（Top 20）
    vector_results = collection.query(
        query_embeddings=[query_vec],
        n_results=20
    )

    # キーワード検索（Top 20）
    keyword_results = collection.query(
        query_texts=[rewritten_query],
        n_results=20
    )

    # 両方をprefLabel単位で統合（vector優先）
    grouped = {}
    id_lookup = {}

    def merge_results(results):
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            pref = meta.get("prefLabel", "")
            if is_filtered(pref, req.granularity):
                continue
            if pref not in grouped:
                grouped[pref] = doc
                id_lookup[pref] = results["ids"][0][i]

    merge_results(vector_results)
    merge_results(keyword_results)

    # 上位5件
    top_contexts = []
    references = []
    for i, (pref, doc) in enumerate(grouped.items()):
        if i >= 5:
            break
        top_contexts.append(doc)
        references.append({
            "id": id_lookup[pref],
            "text": doc[:200] + "...",
            "prefLabel": pref
        })

    context = "\n---\n".join(top_contexts)

    # プロンプト構築
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

    prompt = (
        f"{style}\n\n"
        f"# 質問:\n{req.query}\n\n"
        f"# 法令条文情報:\n{context}"
    )

    # GPT応答
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "references": references
    }

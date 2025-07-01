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

client = OpenAI(api_key=openai_api_key)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="alctax-act")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエスト定義
class AskRequest(BaseModel):
    query: str
    kazemode: bool = False
    granularity: str = "gou"  # "jou", "kou", "gou", "all"

class AskResponse(BaseModel):
    answer: str
    references: list

# 粒度に応じたprefLabelフィルタ
def is_pref_label_valid(pref_label: str, granularity: str) -> bool:
    if granularity == "all":
        return True
    if granularity == "jou":
        return bool(re.fullmatch(r".*第[^条]+条$", pref_label))
    if granularity == "kou":
        return bool(re.fullmatch(r".*第[^条]+条第[^項]+項$", pref_label))
    if granularity == "gou":
        return bool(re.fullmatch(r".*第[^条]+条第[^項]+項第[^号]+号$", pref_label))
    return False

@app.post("/ask", response_model=AskResponse)
async def ask_law(req: AskRequest):
    # ベクトル化
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    )
    query_vec = embed.data[0].embedding

    # Chroma検索
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=30
    )

    grouped = defaultdict(list)
    id_lookup = {}

    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        pref = meta.get("prefLabel", "")
        if not is_pref_label_valid(pref, req.granularity):
            continue
        grouped[pref].append(doc)
        if pref not in id_lookup:
            id_lookup[pref] = results["ids"][0][i]

    top_contexts = []
    references = []

    for i, (pref, docs) in enumerate(grouped.items()):
        if i >= 5:
            break
        combined = "\n".join(docs)
        top_contexts.append(combined)
        references.append({
            "id": id_lookup[pref],
            "text": combined[:200] + "...",
            "prefLabel": pref
        })

    context = "\n---\n".join(top_contexts)

    style = (
        "あなたは藤井風本人です。"
        "以下の法令条文情報【のみ】に基づいて答えてください。"
        "例えや雑談を交えながら、口語体で、リラックスした雰囲気で答えてください。"
    ) if req.kazemode else (
        "あなたは法律の専門家です。"
        "以下の法令条文情報【のみ】に基づいて、簡潔かつ正確に答えてください。"
        "一般常識や事前知識には頼らず、必ず条文内容を根拠にしてください。"
    )

    prompt = f"{style}\n\n# 質問:\n{req.query}\n\n# 法令条文情報:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "references": references
    }

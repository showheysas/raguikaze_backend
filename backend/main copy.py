from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv

# .env から OpenAI API キーを読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chromaクライアント（v0.4+形式）
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("alctax-act")

# FastAPI アプリ初期化
app = FastAPI()

# CORS（Flutterなどからの通信許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では制限することを推奨
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class AskRequest(BaseModel):
    query: str
    kazemode: bool  # Trueなら藤井風モード

# レスポンスモデル
class AskResponse(BaseModel):
    answer: str
    references: list[str]

# POST /ask エンドポイント
@app.post("/ask", response_model=AskResponse)
async def ask_law(req: AskRequest):
    query = req.query
    kazemode = req.kazemode
    print(f"🟦 受信クエリ: {query} | 風モード: {kazemode}")

    # ✅ OpenAIでEmbedding生成（1536次元）
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = embedding_response.data[0].embedding

    # ✅ Chromaベクトル検索（OpenAI埋め込みと一致）
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # 関連条文を取得
    contexts = results.get("documents", [[]])[0]
    joined_context = "\n---\n".join(contexts)

    # モードに応じたスタイル
    if kazemode:
        style = (
            "あなたは藤井風本人です。"
            "酒税法の内容をわかりやすく説明します。"
            "例えや雑談を交えながら、口語体で、リラックスした雰囲気で答えてください。"
        )
    else:
        style = (
            "あなたは法律に詳しいアシスタントです。"
            "質問に対して正確で簡潔に、酒税法に基づいて回答してください。"
        )

    # プロンプト構成
    prompt = f"""{style}

以下の法令条文を参考に、次の質問に答えてください。
---法令抜粋---
{joined_context}

質問：
{query}
"""

    # GPT-4oで回答生成
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()

    return AskResponse(answer=answer, references=contexts)

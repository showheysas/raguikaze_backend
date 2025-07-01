# Python 3.11 ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 必要なパッケージと日本語処理関連ライブラリ
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt を先にコピーして依存解決
COPY requirements.txt .

# pipでインストール（キャッシュを無効化して軽量化）
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースを全てコピー
COPY . .

# 環境変数（例：.env を読み取る場合はRenderで設定も推奨）
ENV CHROMA_DB_PATH=/app/chroma_db
ENV OPENAI_API_KEY=sk-...

# ポートを開放（FastAPI用）
EXPOSE 8000

# アプリケーションの起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#ベースイメージ
FROM python:3.11.4
#コンテナ内の作業ディレクトリ
WORKDIR /app
COPY requirements.txt .
#依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Streamlit の設定（Docker環境でエラーを避ける）
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# コンテナ内で Streamlit を起動
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
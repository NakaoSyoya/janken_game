# ✊✌️🖐️ じゃんけん画像認識AI

手の画像をアップロードすると、AIが「グー・チョキ・パー」を判定し、勝敗を表示するWebアプリケーションです。  
Streamlitを用いてUIを構築し、PyTorchで学習した画像分類モデルをバックエンドで動作させています。  
Dockerコンテナ上でも動作し、ライブラリをローカルにインストールせずとも動きます。

---

## 🚀 デモ
![alt text](<デモ1.png>)
![alt text](<デモ2>.png>)
![alt text](<デモ3.png>)


## 🧠 特徴
- ResNet50をベースにした転移学習モデル（PyTorch）
- KaggleのRock-Paper-Scissorsデータセットを使用
- テストデータ精度：**Accuracy 95%以上**
- Dockerによるコンテナ化で、環境依存を簡易化

---

## 🛠️ 使用技術（Tech Stack）

| カテゴリ | 技術 |
|-----------|------|
| フレームワーク | Streamlit |
| モデル構築 | PyTorch, torchvision |
| モデル構造 | ResNet50（fine-tuning） |
| デプロイ環境 | Docker |
| 開発環境 | Python 3.11.4 |

---

## 🧩 システム構成イメージ
ユーザー → Streamlit UI → 推論API（app.py） → PyTorchモデル → 判定結果表示

---

## 📦 フォルダ構成
janken_game/
├─ app.py # アプリ本体
├─ create_model.py # モデル学習用スクリプト
├─ model/
│ └─ best_model.pth # 学習済みモデル
├─ requirements.txt # 依存ライブラリ
├─ archive/ # Kaggleデータ格納用ディレクトリ
├─デモ1.png
├─デモ2.png
├─デモ3.png
└─ README.md

---

## 🧰 実行方法

### 1. データセット準備
Kaggleから以下のデータをダウンロードして配置してください(しなくてもサイト自体は動きます)：
👉 https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset

ディレクトリ構成：
archive/
└─ Rock-Paper-Scissors/
├─ train/
├─ validation/
└─ test/

ローカルで実行する場合：
### 2. ライブラリをインストール
```bash
pip install -r requirements.txt
```
### 3. アプリを起動
```bash
streamlit run app.py
```
または、Docker環境で動作確認する場合：
### 2. イメージをビルド後、コンテナ起動
```bash
docker build -t janken-streamlit .
docker run -p 8501:8501 janken-streamlit
```

### 3. サイトにアクセス
→ ブラウザで http://localhost:8501 を開く

## 📚create_model.pyを動かすには
### 1. Kaggleのデータセットをダウンロード
訓練データ、検証データが必要なため、この作業は必須となります
### 2.コード実行
```bash
python create_model.py
```
を実行してください。ローカルで実行する場合は事前にライブラリのインストールが必須です。
requirements.txtを参照してライブラリのインストールを行ってください。

## 🧠 モデル概要
ベースモデル：ResNet50

入力画像サイズ：224×224（RGB）

出力クラス：rock, paper, scissors

保存形式：PyTorchモデル（.pth）

## 💡 工夫した点・苦労した点
Kaggleデータセットをベースに、独自にデータ前処理と精度改善を実施

RGB / RGBAなど画像形式の差異を吸収するよう変換処理を実装

Dockerで環境を完全再現し、他PCでも動作可能に

## 🚧 今後の改善ポイント
カメラ入力によるリアルタイムじゃんけん

スマホ対応のレスポンシブUI

モデル軽量化

自動デプロイ（GitHub Actions × DockerHub）

## 📚 参考文献
『Kaggleに挑む深層学習プログラミングの極意』

Kaggle Dataset: Rock-Paper-Scissors

## 👤 制作意図
画像分類モデルをWebで動かしてみたくて作りました。
学習からアプリ化、Dockerでの環境構築まで一通り体験することを目的にしています。
学習済みモデルをファイルに保管する技術があることを初めて知り、感動しました。

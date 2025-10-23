import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import torchvision.models as models
import io

# クラス名
CLASSES = ["paper", "rock", "scissors"]

# モデル読み込み
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

# 保存された重みを読み込む
state_dict = torch.load("model/best_model.pth", map_location="cpu")
model.load_state_dict(state_dict)

# 評価モードへ
model.eval()

# 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASSES[predicted.item()]


def judge(player, ai):
    if player == ai:
        return "引き分け 🤝"
    elif (player == "rock" and ai == "scissors") or \
         (player == "scissors" and ai == "paper") or \
         (player == "paper" and ai == "rock"):
        return "あなたの勝ち 🎉"
    else:
        return "あなたの負け 😢"


st.title("✊✌️🖐️ じゃんけん画像認識AI")

# セッション状態の初期化
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "player_hand" not in st.session_state:
    st.session_state.player_hand = None
if "ai_hand" not in st.session_state:
    st.session_state.ai_hand = None

# ファイルアップロード部分
uploaded_file = st.file_uploader("手の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

# 新しい画像をアップロードした場合は状態更新
if uploaded_file is not None:
    st.session_state.img_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(st.session_state.img_bytes))

    st.image(image, caption="あなたの手", use_container_width=True)

    # 画像がある状態で「AIに判定させる」ボタンを追加
    if st.button("AIに判定させる！"):
        player_hand = predict(image)
        ai_hand = random.choice(CLASSES)
        st.session_state.player_hand = player_hand
        st.session_state.ai_hand = ai_hand
        st.rerun()

# 画像がアップロード済みなら表示
elif st.session_state.img_bytes:
    image = Image.open(io.BytesIO(st.session_state.img_bytes))
    st.image(image, caption="あなたの手", use_container_width=True)

# 結果表示
if st.session_state.player_hand:
    st.write(f"AIの判定結果：**{st.session_state.player_hand}**")
    st.write(f"AIの出した手：**{st.session_state.ai_hand}**")
    st.subheader(judge(st.session_state.player_hand, st.session_state.ai_hand))

    # 🔁 もう一度挑戦ボタン
    if st.button("🔁 もう一度挑戦"):
        st.session_state.player_hand = None
        st.session_state.ai_hand = None
        st.rerun()
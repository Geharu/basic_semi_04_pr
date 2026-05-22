import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

# --- 1. アプリのレイアウト設定 ---
st.title("AI画像判定ツール（ResNet50）")
st.write("スマホで撮った画像をAIが判定します。わざと誤判定させてみよう！")

# --- 2. AIモデルの準備（キャッシュして高速化） ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    # ラベルのダウンロード
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = urllib.request.urlopen(url)
    labels = json.loads(response.read().decode())
    return model, labels

model, labels = load_model()

# 画像変換ルール
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. スマホのカメラ・ファイルアップローダー ---
uploaded_file = st.camera_input("カメラで撮影") # これだけでスマホのカメラが起動します

if uploaded_file is not None:
    # 画像の表示
    img = Image.open(uploaded_file)
    
    # AI判定
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # --- 4. 結果表示 ---
    st.subheader("【AIの判定結果】")
    for i in range(5):
        score = top5_prob[i].item() * 100
        name = labels[top5_catid[i].item()]
        st.write(f"第{i+1}位: **{name}** ({score:.1f}%)")
        st.progress(int(score)) # 視覚的にわかりやすくバーを表示
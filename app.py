import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os
import zipfile

# ----- 設定 -----
class_names = ['glow', 'non_glow']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
BATCH_SIZE = 4
EPOCHS = 3

# ----- モデルロード -----
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("glow_model.pth", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# ----- モード切替 -----
st.sidebar.title("モード選択")
mode = st.sidebar.radio("処理を選んでください", ("単体画像で判定", "ZIP画像を一括判定"))

# ================================
# モード①：単体画像で判定＋フィードバック
# ================================
if mode == "単体画像で判定":
    st.title("Bearing Glow 判定＆フィードバック（単体）")
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロード画像", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            result = class_names[predicted.item()]

        st.subheader(f"判定結果: :blue[{result}]")

        # フィードバック機能
        col1, col2 = st.columns(2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result}_{timestamp}.jpg"

        if col1.button("◯ 合ってた（正解）"):
            st.success("正解として記録されました！")

        if col2.button("× 違ってた（誤判定）"):
            corrected_label = "non_glow" if result == "glow" else "glow"
            save_dir = os.path.join("corrected_data", corrected_label)
            os.makedirs(save_dir, exist_ok=True)
            image.save(os.path.join(save_dir, filename))
            st.warning(f"誤判定画像を保存しました: {filename}")

# ================================
# モード②：ZIPファイルを一括判定＆仕分け
# ================================
elif mode == "ZIP画像を一括判定":
    st.title("Bearing Glow 判定＆仕分け（ZIP一括）")
    uploaded_file = st.file_uploader("ZIPファイルをアップロード", type=["zip"])

    if uploaded_file:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall("uploaded_images")

        image_files = [f for f in os.listdir("uploaded_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            st.error("ZIPファイルに画像が含まれていません。")
        else:
            st.success(f"{len(image_files)} 枚の画像がアップロードされました！")

            result_dir = "sorted_images"
            os.makedirs(os.path.join(result_dir, "glow"), exist_ok=True)
            os.makedirs(os.path.join(result_dir, "non_glow"), exist_ok=True)

            for image_file in image_files:
                image_path = os.path.join("uploaded_images", image_file)
                image = Image.open(image_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    result = class_names[predicted.item()]

                save_path = os.path.join(result_dir, result, image_file)
                image.save(save_path)
                st.image(image, caption=f"{image_file} - 判定結果: {result}", use_column_width=True)

            st.success("すべての画像が仕分けされました！")

# ================================
# 共通：再学習機能
# ================================
st.markdown("---")
if st.button("🔁 フィードバックデータで再学習"):

    data_dir = "corrected_data"
    if not os.path.exists(data_dir):
        st.error("再学習用のフィードバック画像がありません。")
    else:
        st.info("再学習を開始します...")

        dataset = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        finetune_model = models.resnet18(pretrained=False)
        finetune_model.fc = nn.Linear(finetune_model.fc.in_features, 2)
        finetune_model.load_state_dict(torch.load("glow_model.pth", map_location=device))
        finetune_model = finetune_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(finetune_model.parameters(), lr=1e-4)

        finetune_model.train()
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = finetune_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            st.write(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")

        torch.save(finetune_model.state_dict(), "glow_model.pth")
        st.success("再学習完了！モデルを更新しました ✅")

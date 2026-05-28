import pandas as pd
import torch
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.career_dataset import CareerDataset
from src.transformer_model import CareerTransformer
from src.trainer import Trainer

# 1. Định nghĩa FEATURES bằng chữ thường để đồng bộ
FEATURES = [
    'toan', 'van', 'anh', 'ly', 'hoa', 'sinh', 'su', 'dia', 'gdcd', 
    'khtn', 'khxh', 'r', 'i', 'a', 's', 'e', 'c'
]

# 2. LOAD DATA
df = pd.read_csv("data/synthetic_students.csv")

# CHUẨN HÓA TÊN CỘT: Xóa khoảng trắng, chuyển về chữ thường hết
df.columns = df.columns.str.strip().str.lower()

# 3. KIỂM TRA VÀ TRÍCH XUẤT DỮ LIỆU
# Lấy đúng 17 cột theo thứ tự của FEATURES
try:
    X_raw = df[FEATURES]
    y_raw = df["major"]
except KeyError as e:
    print(f"❌ Lỗi: Không tìm thấy cột trong file CSV: {e}")
    print(f"Các cột thực tế đang có là: {df.columns.tolist()}")
    exit()

# 4. ENCODE LABEL (Xử lý các ngành tiếng Việt có dấu)
le = LabelEncoder()
y = le.fit_transform(y_raw)

if not os.path.exists("models"):
    os.makedirs("models")

with open("models/classes.pkl", "wb") as f:
    pickle.dump(le.classes_, f)

# 5. SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 6. DATASET & MODEL
dataset = CareerDataset(X_scaled, pd.Series(y))
num_classes = len(le.classes_)

# Khởi tạo model với input_dim=17
model = CareerTransformer(
    input_dim=len(FEATURES), 
    num_classes=num_classes
)

# 7. TRAIN
print(f"Starting training with {num_classes} majors...")
trainer = Trainer(model, dataset)
trainer.train(epochs=100) 

# 8. SAVE MODEL
torch.save(model.state_dict(), "models/career_model.pth")
print("✅ Thành công! Model và Scaler đã khớp hoàn toàn 17 features.")
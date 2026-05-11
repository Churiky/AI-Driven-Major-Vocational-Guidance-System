import torch
import torch.nn.functional as F
import pickle
import numpy as np

from src.personality_test import HollandTest
from src.holland_mapper import HollandMapper
from src.transformer_model import CareerTransformer
from src.university_recommender import UniversityRecommender
from src.admission_predictor import AdmissionPredictor

# =========================
# LOAD CONFIG & LABELS (DÙNG CHO DYNAMIC)
# =========================
try:
    with open("models/classes.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model hoặc nhãn. Vui lòng chạy train_model.py trước!")
    exit()

num_classes = len(labels)

# =========================
# HOLLAND TEST & INPUT SCORE
# =========================
# (Giữ nguyên phần nhập liệu Holland và điểm số của bạn...)
print("\n===== HOLLAND TEST =====")
test = HollandTest()
holland = test.run()

print("\n===== CHỌN KHỐI THI =====")
choice = input("Chọn (1. Tự nhiên / 2. Xã hội): ")
toan = float(input("Toán: "))
van = float(input("Văn: "))
anh = float(input("Anh: "))
ly = hoa = sinh = su = dia = gdcd = 0

if choice == "1":
    ly, hoa, sinh = float(input("Lý: ")), float(input("Hóa: ")), float(input("Sinh: "))
else:
    su, dia, gdcd = float(input("Sử: ")), float(input("Địa: ")), float(input("GDCD: "))

scores = {
    "toan": toan, "van": van, "anh": anh,
    "ly": ly, "hoa": hoa, "sinh": sinh,
    "su": su, "dia": dia, "gdcd": gdcd,
    "khtn": (ly + hoa + sinh) / 3 if choice == "1" else 0,
    "khxh": (su + dia + gdcd) / 3 if choice == "2" else 0,
    "ngoai_ngu": anh
}

# =========================
# BUILD FEATURE VECTOR
# =========================
vector_raw = [
    toan, van, anh, ly, hoa, sinh, su, dia, gdcd,
    (ly+hoa+sinh)/3, (su+dia+gdcd)/3,
    holland["R"], holland["I"], holland["A"], holland["S"], holland["E"], holland["C"]
]

# Chuẩn hóa vector bằng scaler đã dùng khi train
vector_scaled = scaler.transform([vector_raw])
vector_tensor = torch.tensor(vector_scaled, dtype=torch.float32)

# =========================
# LOAD MODEL & PREDICT
# =========================
model = CareerTransformer(input_dim=17, num_classes=num_classes)
model.load_state_dict(torch.load("models/career_model.pth"))
model.eval()

with torch.no_grad():
    output = model(vector_tensor)
    probs = F.softmax(output, dim=1)

# Lấy Top 3 ngành có xác suất cao nhất
top_probs, top_idxs = torch.topk(probs, k=min(3, num_classes))

print("\n===== KẾT QUẢ DỰ ĐOÁN AI =====")
for i in range(top_probs.size(1)):
    idx = top_idxs[0][i].item()
    p = top_probs[0][i].item()
    print(f"Top {i+1}: {labels[idx]} ({round(p*100, 2)}%)")

# Chọn ngành cao nhất để gợi ý trường
major = labels[top_idxs[0][0].item()]

# =========================
# UNIVERSITY RECOMMEND
# =========================
recommender = UniversityRecommender("data/diem_chuan_all.csv")
results = recommender.recommend(major, scores)
predictor = AdmissionPredictor()

print(f"\n===== TRƯỜNG GỢI Ý CHO NGÀNH: {major} =====")
if not results:
    print("Không tìm thấy trường phù hợp.")
else:
    for r in results[:10]: # Hiện top 10 trường
        prob_val = predictor.predict(r["Score"], r["Cutoff"])
        print(f"{r['Truong']} | {r['Nganh']} | Khối: {r['Block']} | Điểm: {r['Score']} | Điểm chuẩn: {r['Cutoff']} | {r['DanhGia']} | Đậu: {prob_val}%")
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import torch
import torch.nn.functional as F
import pickle
import os
import threading
import numpy as np

from src.transformer_model import CareerTransformer, FeatureTokenizerTransformer, MultimodalCareerTransformer
from src.university_recommender import UniversityRecommender
from src.admission_predictor import AdmissionPredictor
from src.ai_expert import CareerAIExpert 

app = Flask(__name__)
app.secret_key = "career_ai_secret_key"

# ==========================================
# 🔑 CẤU HÌNH API KEY (ĐÃ SỬA LOGIC KHỞI TẠO)
# ==========================================
GEMINI_API_KEY = "AIzaSyCqo5MWq-zM3BBlqTYbUdP5oFnEzrHWfN8"

MAJOR_MAPPING = {
    "CongNgheThongTin": "Công Nghệ Thông Tin",
    "YKhoa": "Y Khoa",
    "KinhTe": "Kinh Tế",
    "NgonNguAnh": "Ngôn Ngữ Anh",
    "Luat": "Luật",
    "KỹThuatOTo": "Kỹ Thuật Ô Tô",
    "KienTruc": "Kiến Trúc",
    "Marketing": "Marketing",
    "TamLyHoc": "Tâm Lý Học",
    "SuPham": "Sư Phạm",
    "QuanTriKhachSan": "Quản Trị Khách Sạn",
    "KeToan": "Kế Toán"
}

HOLLAND_INFO = {
    "R": {
        "label": "Thực tế",
        "title": "Nhóm R",
        "description": "Thích thao tác, vận hành, sửa chữa và làm việc với hệ thống cụ thể.",
    },
    "I": {
        "label": "Nghiên cứu",
        "title": "Nhóm I",
        "description": "Mạnh về phân tích, tìm hiểu, quan sát và giải quyết vấn đề.",
    },
    "A": {
        "label": "Nghệ thuật",
        "title": "Nhóm A",
        "description": "Có xu hướng sáng tạo, thiết kế, trình bày ý tưởng và cảm nhận thẩm mỹ.",
    },
    "S": {
        "label": "Xã hội",
        "title": "Nhóm S",
        "description": "Nổi trội ở khả năng hỗ trợ, lắng nghe, hướng dẫn và làm việc với con người.",
    },
    "E": {
        "label": "Quản lý",
        "title": "Nhóm E",
        "description": "Phù hợp với lãnh đạo, tổ chức, kinh doanh, thuyết phục và ra quyết định.",
    },
    "C": {
        "label": "Nghiệp vụ",
        "title": "Nhóm C",
        "description": "Mạnh về quy trình, chi tiết, dữ liệu, sự chính xác và tính hệ thống.",
    },
}

SUBJECT_LABELS = {
    "toan": "Toán",
    "van": "Ngữ Văn",
    "anh": "Tiếng Anh",
    "ly": "Vật Lý",
    "hoa": "Hóa Học",
    "sinh": "Sinh Học",
    "su": "Lịch Sử",
    "dia": "Địa Lí",
    "gdcd": "Giáo Dục Công Dân",
    "khtn": "Khoa Học Tự Nhiên",
    "khxh": "Khoa Học Xã Hội",
}

MAJOR_INSIGHTS = {
    "CongNgheThongTin": {
        "theme": "Công nghệ và dữ liệu",
        "fit_reason": "Phù hợp với tư duy logic, khả năng giải quyết vấn đề và sự bền bỉ khi làm việc với hệ thống.",
        "roles": ["Lập trình viên", "Kỹ sư phần mềm", "Phân tích dữ liệu"]
    },
    "YKhoa": {
        "theme": "Sức khỏe và chăm sóc",
        "fit_reason": "Cần sự tập trung, nền tảng khoa học tự nhiên và xu hướng quan tâm đến con người.",
        "roles": ["Bác sĩ", "Dược sĩ", "Điều dưỡng"]
    },
    "KinhTe": {
        "theme": "Kinh doanh và tài chính",
        "fit_reason": "Phù hợp với người có thế mạnh toán, phân tích và định hướng tổ chức - kinh doanh.",
        "roles": ["Chuyên viên tài chính", "Phân tích kinh doanh", "Quản trị doanh nghiệp"]
    },
    "NgonNguAnh": {
        "theme": "Ngôn ngữ và giao tiếp",
        "fit_reason": "Phát huy khả năng ngôn ngữ, giao tiếp, diễn đạt và làm việc trong môi trường hội nhập.",
        "roles": ["Biên phiên dịch", "Giảng dạy tiếng Anh", "Truyền thông quốc tế"]
    },
    "Luat": {
        "theme": "Pháp lý và quy định",
        "fit_reason": "Phù hợp với người có tư duy lập luận, cẩn thận và quan tâm đến quy tắc - hệ thống.",
        "roles": ["Chuyên viên pháp chế", "Tư vấn pháp lý", "Kiểm soát tuân thủ"]
    },
    "KyThuatOTo": {
        "theme": "Kỹ thuật và vận hành",
        "fit_reason": "Nổi trội ở khả năng thao tác, sửa chữa, kết hợp giữa toán - lý và tính thực hành.",
        "roles": ["Kỹ sư ô tô", "Cố vấn dịch vụ kỹ thuật", "Quản lý bảo trì"]
    },
    "KienTruc": {
        "theme": "Thiết kế và không gian",
        "fit_reason": "Cần sự cân bằng giữa sáng tạo, cảm nhận thẩm mỹ và nền tảng logic kỹ thuật.",
        "roles": ["Kiến trúc sư", "Thiết kế nội thất", "Quy hoạch không gian"]
    },
    "Marketing": {
        "theme": "Truyền thông và thương hiệu",
        "fit_reason": "Phù hợp với giao tiếp, sáng tạo nội dung và định hướng thị trường - khách hàng.",
        "roles": ["Chuyên viên marketing", "Brand executive", "Nội dung số"]
    },
    "TamLyHoc": {
        "theme": "Tâm lý và hỗ trợ con người",
        "fit_reason": "Nổi trội khi kết hợp lắng nghe, đồng cảm, quan sát và phân tích hành vi.",
        "roles": ["Tham vấn tâm lý", "Tư vấn học đường", "Nghiên cứu hành vi"]
    },
    "SuPham": {
        "theme": "Giáo dục và hướng dẫn",
        "fit_reason": "Phù hợp với người thích chia sẻ kiến thức, hỗ trợ người khác và làm việc có cấu trúc.",
        "roles": ["Giáo viên", "Chuyên viên giáo dục", "Đào tạo nội bộ"]
    },
    "QuanTriKhachSan": {
        "theme": "Dịch vụ và vận hành",
        "fit_reason": "Cần khả năng giao tiếp, tổ chức, linh hoạt và xử lý trải nghiệm khách hàng.",
        "roles": ["Quản lý khách sạn", "Điều phối dịch vụ", "Vận hành du lịch"]
    },
    "KeToan": {
        "theme": "Tài chính và kiểm soát",
        "fit_reason": "Phù hợp với người cẩn thận, logic, yêu thích con số và quy trình rõ ràng.",
        "roles": ["Kế toán viên", "Kiểm toán viên", "Chuyên viên kiểm soát nội bộ"]
    }
}


def build_holland_profile(holland_scores):
    sorted_codes = sorted(holland_scores.items(), key=lambda item: item[1], reverse=True)
    dominant = []
    for code, score in sorted_codes[:3]:
        info = HOLLAND_INFO.get(code, {})
        dominant.append(
            {
                "code": code,
                "score": score,
                "label": info.get("label", code),
                "title": info.get("title", code),
                "description": info.get("description", ""),
            }
        )

    profile_name = " + ".join(item["code"] for item in dominant[:2]) if dominant else "RIASEC"
    return dominant, profile_name


def build_subject_strengths(scores, limit=4):
    ranked = []
    for key in ["toan", "van", "anh", "ly", "hoa", "sinh", "su", "dia", "gdcd"]:
        score = float(scores.get(key, 0))
        ranked.append(
            {
                "key": key,
                "label": SUBJECT_LABELS.get(key, key),
                "score": score,
                "width": min(max(score * 10, 0), 100),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def build_academic_profile(scores):
    tracks = [
        ("KHTN", float(scores.get("khtn", 0))),
        ("KHXH", float(scores.get("khxh", 0))),
        ("Ngoai ngu", float(scores.get("anh", 0))),
    ]
    tracks.sort(key=lambda item: item[1], reverse=True)
    return tracks


def build_major_cards(top_predictions):
    cards = []
    for prediction in top_predictions:
        insight = MAJOR_INSIGHTS.get(prediction["raw"], {})
        cards.append(
            {
                "name": prediction["name"],
                "confidence": prediction["p"],
                "theme": insight.get("theme", "Định Hướng Nghề Nghiệp"),
                "fit_reason": insight.get("fit_reason", "Phù Hợp Với Hồ Sơ Hiện Tại Của Học Sinh."),
                "roles": insight.get("roles", ["Vị Trí Liên Quan"]),
            }
        )
    return cards


def build_dashboard_payload(holland_scores, scores, top_predictions, recommendations):
    dominant_holland, profile_name = build_holland_profile(holland_scores)
    subject_strengths = build_subject_strengths(scores)
    academic_tracks = build_academic_profile(scores)
    major_cards = build_major_cards(top_predictions)

    total_matches = len(recommendations)
    safe_matches = sum(1 for item in recommendations if float(item.get("Prob", 0)) >= 90)
    balanced_matches = sum(
        1 for item in recommendations if 30 <= float(item.get("Prob", 0)) < 90
    )
    stretch_matches = max(total_matches - safe_matches - balanced_matches, 0)
    top_school = recommendations[0] if recommendations else None
    avg_cutoff = round(
        sum(item["Cutoff"] for item in recommendations[:10]) / min(len(recommendations), 10), 1
    ) if recommendations else 0

    dashboard_data = {
        "profile_name": profile_name,
        "dominant_holland": dominant_holland,
        "subject_strengths": subject_strengths,
        "academic_tracks": [
            {"label": label, "score": round(score, 2), "width": min(max(score * 10, 0), 100)}
            for label, score in academic_tracks
        ],
        "stats": {
            "total_matches": total_matches,
            "safe_matches": safe_matches,
            "balanced_matches": balanced_matches,
            "stretch_matches": stretch_matches,
            "avg_cutoff": avg_cutoff,
            "best_probability": top_predictions[0]["p"] if top_predictions else 0,
        },
        "top_school": top_school,
    }

    career_map = {
        "profile_nodes": dominant_holland,
        "subject_nodes": subject_strengths[:3],
        "major_nodes": major_cards,
        "target_schools": [
            {
                "school": item["Truong"],
                "major": item["Nganh"],
                "block": item["Block"],
                "probability": item["Prob"],
                "label": item["DanhGia"],
            }
            for item in recommendations[:3]
        ],
    }

    return dashboard_data, career_map

# ==========================================
# 1. LOAD AI MODEL & CONFIG
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
classes_path = os.path.join(base_dir, "models", "multimodal_transformer_classes.pkl")
scaler_path = os.path.join(base_dir, "models", "multimodal_transformer_scaler.pkl")
model_path = os.path.join(base_dir, "models", "multimodal_transformer.pth")

with open(classes_path, "rb") as f: labels = pickle.load(f)
with open(scaler_path, "rb") as f: scaler = pickle.load(f)

num_classes = len(labels)


def load_prediction_model(path, num_classes):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("config", {})
        model = MultimodalCareerTransformer(
            num_acad_features=11,
            num_psych_features=6,
            num_classes=num_classes,
            d_model=64,
            nhead=4,
            num_layers=3,
            ff_multiplier=3,
            dropout=0.15
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    model = CareerTransformer(input_dim=17, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    return model


if os.path.exists(model_path):
    model = load_prediction_model(model_path, num_classes)
else:
    model = CareerTransformer(input_dim=17, num_classes=num_classes)

model.eval()

recommender = UniversityRecommender(os.path.join(base_dir, "data", "diem_chuan_all.csv"))
predictor = AdmissionPredictor()

# LAZY INIT RAG EXPERT
expert_rag = None
expert_rag_lock = threading.Lock()
expert_rag_init_error = None


def get_rag_expert():
    global expert_rag
    global expert_rag_init_error

    if expert_rag is not None:
        return expert_rag

    if not GEMINI_API_KEY or len(GEMINI_API_KEY) <= 10:
        expert_rag_init_error = "Chưa cấu hình Gemini API Key hợp lệ."
        return None

    with expert_rag_lock:
        if expert_rag is not None:
            return expert_rag

        try:
            pdf_dir = os.path.join(base_dir, "data", "documents")
            print(f"Initializing RAG with document directory: {pdf_dir}")
            expert_rag = CareerAIExpert(api_key=GEMINI_API_KEY, pdf_path=pdf_dir)
            expert_rag_init_error = None
            print("RAG Expert is ready on the website.")
            return expert_rag
        except Exception as e:
            expert_rag = None
            expert_rag_init_error = str(e)
            print(f"RAG init error: {e}")
            return None

# ==========================================
# 2. ROUTES 
# ==========================================

@app.route("/", methods=["GET", "POST"])
def holland():
    if request.method == "POST":
        session['holland'] = {
            "R": int(request.form.get("R", 0)),
            "I": int(request.form.get("I", 0)),
            "A": int(request.form.get("A", 0)),
            "S": int(request.form.get("S", 0)),
            "E": int(request.form.get("E", 0)),
            "C": int(request.form.get("C", 0))
        }
        return redirect(url_for('score'))
    return render_template("holland.html")

@app.route("/score", methods=["GET", "POST"])
def score():
    if request.method == "POST":
        s = {
            "toan": float(request.form.get("toan", 0)),
            "van": float(request.form.get("van", 0)),
            "anh": float(request.form.get("anh", 0)),
            "ly": float(request.form.get("ly", 0)),
            "hoa": float(request.form.get("hoa", 0)),
            "sinh": float(request.form.get("sinh", 0)),
            "su": float(request.form.get("su", 0)),
            "dia": float(request.form.get("dia", 0)),
            "gdcd": float(request.form.get("gdcd", 0)),
            "method_id": int(request.form.get("method_id", 1))
        }
        s["khtn"] = round((s["ly"] + s["hoa"] + s["sinh"]) / 3, 2)
        s["khxh"] = round((s["su"] + s["dia"] + s["gdcd"]) / 3, 2)
        s["ngoai_ngu"] = s["anh"]
        
        session['scores'] = s
        return redirect(url_for('preference'))
    return render_template("score.html")
@app.route("/preference", methods=["GET", "POST"])
def preference():
    if request.method == "POST":
        # Lưu các tùy chọn vào session
        session['preferences'] = {
            "priority_order": request.form.get("priority_order"),
            "region": request.form.get("preferred_region"),
            "city": request.form.get("preferred_city"),
            "type": request.form.get("preferred_type")
        }
        return redirect(url_for('result'))
    return render_template("preference.html")
@app.route("/result")
def result():
    if 'holland' not in session or 'scores' not in session or 'preferences' not in session:
        return redirect(url_for('holland'))

    h = session['holland']
    s = session['scores']
    input_vector = [
        s['toan'], s['van'], s['anh'], s['ly'], s['hoa'], s['sinh'], 
        s['su'], s['dia'], s['gdcd'], s['khtn'], s['khxh'],
        h['R'], h['I'], h['A'], h['S'], h['E'], h['C']
    ]

    input_scaled = scaler.transform([input_vector])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
    
    top_probs, top_idxs = torch.topk(probs, k=min(3, num_classes))
    display_probs = []
    top_predictions = []
    for i in range(top_probs.size(1)):
        idx = top_idxs[0][i].item()
        p = round(top_probs[0][i].item() * 100, 1)
        major_raw = labels[idx]
        major_pretty = MAJOR_MAPPING.get(major_raw, major_raw)
        prediction = {"raw": major_raw, "name": major_pretty, "p": p}
        top_predictions.append(prediction)
        display_probs.append({"name": major_pretty, "p": p})

    best_major_raw = labels[top_idxs[0][0].item()]
    best_major_pretty = MAJOR_MAPPING.get(best_major_raw, best_major_raw)
    
    # Lấy các tùy chọn ưu tiên từ session (Vùng miền, Tỉnh thành, Loại trường, Thứ tự kéo thả)
    pref = session.get('preferences', {})
    preferred_region = pref.get('region')
    preferred_city   = pref.get('city')
    preferred_type   = pref.get('type')
    priority_order   = pref.get('priority_order')
    
    expand_majors = request.args.get('expand', '0') == '1'
    
    preferred_method = s.get('method_id', 1)
    
    raw_results = recommender.recommend(
        best_major_raw, s, 
        preferred_region=preferred_region,
        preferred_city=preferred_city,
        preferred_type=preferred_type,
        priority_order=priority_order,
        expand_majors=expand_majors,
        preferred_method=preferred_method
    )
    dashboard_data, career_map = build_dashboard_payload(h, s, top_predictions, raw_results)

    return render_template("result.html", 
                           major=best_major_pretty, 
                           display_probs=display_probs, 
                           results=raw_results,
                           expand_majors=expand_majors,
                           holland_data=h,
                           dashboard_data=dashboard_data,
                           career_map=career_map)

# ==========================================
# 3. EXPERT CHATBOT API (FIXED)
# ==========================================
@app.route("/api/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")

    rag = get_rag_expert()

    if rag is not None:
        try:
            reply = rag.chat_with_admission(user_msg)
            return jsonify({"reply": reply})
        except Exception as e:
            print(f"Chat error: {e}")

    # Fallback chi tiết hơn
    msg = user_msg.lower()
    if "hoc phi" in msg or "tiền" in msg:
        reply = "Theo thông báo tuyển sinh 2025, học phí trung bình khoảng 15-40 triệu/năm tùy hệ đào tạo."
    elif "xet tuyen" in msg or "phương thức" in msg:
        reply = "Đại học Bách khoa TP.HCM xét tuyển chủ yếu qua: Điểm thi ĐGNL, Xét tuyển tổng hợp và Tuyển thẳng."
    else:
        if expert_rag_init_error:
            reply = f"Tôi chưa thể khởi tạo hệ thống RAG lúc này. Chi tiết lỗi: {expert_rag_init_error}"
        else:
            reply = "Tôi đã nhận được câu hỏi. Hệ thống RAG đang khởi động, bạn hãy thử hỏi lại sau ít giây nhé!"
    
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

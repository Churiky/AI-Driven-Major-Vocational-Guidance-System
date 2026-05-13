import pandas as pd
import numpy as np
import re
import os
from unidecode import unidecode
from src.subject_mapper import SubjectMapper
from src.admission_predictor import AdmissionPredictor


class UniversityRecommender:
    def __init__(self, path):
        """
        Khởi tạo hệ thống gợi ý chuyên sâu.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path}")

        # 1. Đọc và chuẩn hóa dữ liệu
        self.df = pd.read_csv(path, encoding='utf-8-sig')
        self.df.columns = self.df.columns.str.strip()

        # 2. Khởi tạo module hỗ trợ
        self.mapper = SubjectMapper()
        self.predictor = AdmissionPredictor()

        # 3. Mapping ngành -> từ khóa tìm kiếm (đầy đủ từ file gốc + bổ sung BaoChi)
        self.mapping = {
            "CongNgheThongTin": [
                "cong nghe thong tin", "cntt", "khoa hoc may tinh",
                "ky thuat phan mem", "he thong thong tin", "an toan thong tin"
            ],
            "KinhTe": [
                "kinh te", "tai chinh", "ke toan", "quan tri",
                "ngan hang", "thuong mai", "kinh doanh"
            ],
            "YKhoa": [
                "y khoa", "y duoc", "bac si", "duoc hoc", "dieu duong", "nha khoa"
            ],
            "NgonNguAnh": [
                "ngon ngu anh", "su pham tieng anh", "phien dich tieng anh"
            ],
            "Luat": [
                "luat", "luat kinh te", "luat thuong mai", "luat quoc te"
            ],
            "KyThuatOTo": [
                "o to", "co khi dong luc", "ky thuat o to", "cong nghe o to"
            ],
            "KienTruc": [
                "kien truc", "quy hoach", "thiet ke noi that"
            ],
            "Marketing": [
                "marketing", "truyen thong", "pr", "quan tri thuong hieu"
            ],
            "TamLyHoc": [
                "tam ly hoc", "tam ly giao duc", "tham van"
            ],
            "SuPham": [
                "su pham", "giao duc mam non", "giao duc tieu hoc"
            ],
            "QuanTriKhachSan": [
                "khach san", "nha hang", "du lich", "quan tri dich vu"
            ],
            "KeToan": [
                "ke toan", "kiem toan"
            ],
            "BaoChi": [
                "bao chi", "tuyen truyen", "truyen thong", "phat thanh", "truyen hinh"
            ],
        }

        # 4. Tiền xử lý thuộc tính cho TOPSIS
        self._prepare_topsis_features()

        # 5. Load điểm chất lượng đào tạo từ data_da_hop_nhat.csv
        self.quality_scores = {}
        self.school_regions = {}
        self.school_types = {}
        quality_path = os.path.join(os.path.dirname(path), 'data_da_hop_nhat.csv')
        if os.path.exists(quality_path):
            try:
                df_quality = pd.read_csv(quality_path, encoding='utf-8-sig')
                if 'Tên trường' in df_quality.columns and 'Tổng điểm' in df_quality.columns:
                    # Chuyển đổi sang string và xóa khoảng trắng để map cho chuẩn
                    df_quality['Tên trường'] = df_quality['Tên trường'].astype(str).str.strip()
                    for _, row_q in df_quality.iterrows():
                        norm_name = self._normalize_school_name(row_q['Tên trường'])
                        self.quality_scores[norm_name] = row_q['Tổng điểm']
                        # Lưu thêm vùng miền nếu có
                        if 'Vị trí địa lý' in df_quality.columns:
                            self.school_regions[norm_name] = str(row_q['Vị trí địa lý']).strip()
                        # Lưu thêm loại trường (Công lập/Tư thục)
                        if 'Loại trường' in df_quality.columns:
                            self.school_types[norm_name] = str(row_q['Loại trường']).strip()
            except Exception as e:
                print(f"Không thể tải điểm chất lượng: {e}")

    # -----------------------------------------------------------------------
    # Tiền xử lý
    # -----------------------------------------------------------------------

    def _normalize_school_name(self, name):
        """Chuẩn hóa tên trường cực mạnh để mapping."""
        if not name or not isinstance(name, str):
            return ""
        # 1. Xóa mã trường (ví dụ: "QSK - ", "LPH - ")
        name = re.sub(r'^[A-Z0-9]+\s*-\s*', '', name)
        # 2. Xóa các thông tin bổ sung trong ngoặc đơn (ví dụ: "(Cơ sở 2)", "(CS2)")
        name = re.sub(r'\s*\(.*?\)\s*', ' ', name)
        # 3. Xóa các từ chung và từ chỉ chi nhánh
        for word in ["Trường Đại học", "Đại học", "Học viện", "Cơ sở", "Phân hiệu"]:
            name = name.replace(word, "")
        
        name = name.replace("TP.", " ").replace("T.P.", " ").replace("TP", " ")
        name = name.replace(".", "").replace(",", "")
        return self._clean_text(name).strip()

    def _prepare_topsis_features(self):
        """Định lượng hóa cột Loại trường và Thành phố để dùng trong TOPSIS."""
        loai_col = self._find_col(['Loại trường', 'loai_truong', 'Loai truong'])
        city_col = self._find_col(['Thành phố', 'tinh_thanh_pho', 'Thanh pho'])

        big_cities = ['Hà Nội', 'TP. Hồ Chí Minh', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng',
                      'Ha Noi', 'Ho Chi Minh', 'Da Nang', 'Can Tho', 'Hai Phong']

        if loai_col:
            self.df['Type_Score'] = self.df[loai_col].apply(
                lambda x: 1.0 if str(x).strip() in ('Công lập', 'Cong lap') else 0.5
            )
        else:
            self.df['Type_Score'] = 0.5

        if city_col:
            self.df['Location_Score'] = self.df[city_col].apply(
                lambda x: 1.0 if any(city in str(x) for city in big_cities) else 0.5
            )
        else:
            self.df['Location_Score'] = 0.5

    def _find_col(self, candidates):
        """Tìm tên cột thực tế trong DataFrame theo danh sách ứng viên."""
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def _clean_text(self, text):
        """Chuẩn hóa text: bỏ dấu, lowercase."""
        if not isinstance(text, str):
            return ""
        return unidecode(text).lower().strip()

    # -----------------------------------------------------------------------
    # Lấy giá trị ô theo nhiều tên cột có thể có
    # -----------------------------------------------------------------------

    def _get(self, row, candidates, default=""):
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return default

    # -----------------------------------------------------------------------
    # Recommend
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # TOPSIS - MCDM Core Logic
    # -----------------------------------------------------------------------

    def _calculate_topsis_weights(self, priority_order):
        """
        Chuyển đổi thứ tự ưu tiên thành bộ trọng số.
        Thứ tự mặc định: Level_Match, Region_Match, Type_Match, City_Match, Quality_Score, Major_Score, Close_Match
        """
        criteria = ['Level_Match', 'Region_Match', 'Type_Match', 'City_Match', 'Quality_Score', 'Major_Score', 'Close_Match']
        weights = {c: 0.05 for c in criteria} # Trọng số mặc định tối thiểu

        if priority_order:
            # Mapping từ key giao diện sang key nội bộ
            key_map = {
                'chat_luong': 'Level_Match',
                'vung_mien': 'Region_Match',
                'loai_truong': 'Type_Match',
                'tinh_thanh': 'City_Match'
            }
            
            # Phân bổ trọng số theo thứ tự (40%, 25%, 15%, 10%)
            priority_weights = [0.40, 0.25, 0.15, 0.10]
            ordered_keys = [k.strip() for k in priority_order.split(',') if k.strip() in key_map]
            
            for i, key in enumerate(ordered_keys):
                if i < len(priority_weights):
                    internal_key = key_map[key]
                    weights[internal_key] = priority_weights[i]

        # Chuẩn hóa để tổng trọng số bằng 1
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}

    def _compute_topsis(self, df, weights):
        """
        Thực hiện các bước toán học của TOPSIS.
        """
        if df.empty:
            return df

        # 1. Chuẩn bị ma trận quyết định (Decision Matrix)
        criteria = list(weights.keys())
        matrix = df[criteria].values.astype(float)

        # 2. Chuẩn hóa ma trận (Vector Normalization)
        # Tránh chia cho 0 nếu một cột có tất cả giá trị bằng 0
        norm = np.sqrt(np.sum(matrix**2, axis=0))
        norm[norm == 0] = 1e-9
        norm_matrix = matrix / norm

        # 3. Nhân trọng số (Weighted Normalization)
        w_array = np.array([weights[c] for c in criteria])
        weighted_matrix = norm_matrix * w_array

        # 4. Xác định giải pháp lý tưởng Tốt (+) và Xấu (-)
        # Tất cả các tiêu chí của chúng ta đều là Benefit (càng cao càng tốt)
        ideal_best = np.max(weighted_matrix, axis=0)
        ideal_worst = np.min(weighted_matrix, axis=0)

        # 5. Tính khoảng cách Euclid tới giải pháp lý tưởng
        dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # 6. Tính điểm Closeness Coefficient (Ci)
        # Ci = dw / (db + dw)
        # Tránh chia cho 0
        total_dist = dist_best + dist_worst
        total_dist[total_dist == 0] = 1e-9
        topsis_scores = dist_worst / total_dist

        df['TOPSIS_Score'] = topsis_scores
        return df

    # -----------------------------------------------------------------------
    # Recommend
    # -----------------------------------------------------------------------

    def recommend(self, nganh_key, user_scores, weights=None, preferred_region=None, preferred_city=None, preferred_type=None, priority_order=None, expand_majors=False):
        """
        Pipeline chính sử dụng thuật toán TOPSIS để xếp hạng.
        """
        # Bảng ánh xạ vùng miền
        region_map = {
            "Miền Bắc": ["Đồng bằng sông Hồng", "Trung du và miền núi phía Bắc"],
            "Miền Trung": ["Bắc Trung Bộ", "Duyên hải Nam Trung Bộ và Tây Nguyên"],
            "Miền Nam": ["Đông Nam Bộ"],
            "Miền Tây": ["Đồng bằng sông Cửu Long"]
        }

        keywords = self.mapping.get(nganh_key, [])
        if not keywords: return []

        col_truong  = self._find_col(['truong', 'Tên trường', 'ten_truong'])
        col_nganh   = self._find_col(['ten_nganh', 'Tên ngành', 'Nganh'])
        col_cutoff  = self._find_col(['diem_chuan', 'Diem chuan'])
        col_tohop   = self._find_col(['to_hop', 'To hop'])

        if not col_cutoff or not col_tohop:
            raise ValueError("CSV thiếu cột điểm chuẩn hoặc tổ hợp môn.")

        recommendations = []

        for _, row in self.df.iterrows():
            # --- BƯỚC 1: LỌC ĐIỂM CHUẨN & NGÀNH ---
            try:
                cutoff = float(row[col_cutoff])
            except (ValueError, TypeError): continue
            if cutoff > 35: continue

            ten_nganh_raw = str(self._get(row, [col_nganh] if col_nganh else [], ""))
            ten_nganh_clean = self._clean_text(ten_nganh_raw)
            
            # Filter by major keyword
            is_match = any(k in ten_nganh_clean for k in keywords)
            if not is_match: continue
            
            major_score = 1.0 if any(k == ten_nganh_clean for k in keywords) else 0.5

            # --- BƯỚC 2: TÍNH ĐIỂM TỔ HỢP ---
            to_hop_raw = str(row[col_tohop])
            sep = ";" if ";" in to_hop_raw else ","
            blocks = [b.strip() for b in to_hop_raw.split(sep)]

            best_score = -1
            valid_block = ""
            for b in blocks:
                if re.match(r'^[A-Z][0-9]{2}[A-Z]?$', b.strip()):
                    score = self.mapper.calculate_score(b.strip(), user_scores)
                    if score is not None and score > best_score:
                        best_score = score
                        valid_block = b

            if best_score == -1: continue
            current_score = best_score
            current_prob = self.predictor.predict(current_score, cutoff)

            # --- BƯỚC 3: CHUẨN BỊ TIÊU CHÍ CHO TOPSIS ---
            diff = current_score - cutoff
            current_danhgia = "An toàn" if diff >= 1.5 else ("Vừa sức" if diff >= -1.5 else "Tỉ lệ thấp")
            
            level_match = 1.0 if current_danhgia in ["An toàn", "Vừa sức"] else 0.2
            close_match = 1.0 if abs(diff) <= 1.5 else 0.1

            ten_truong = str(self._get(row, [col_truong] if col_truong else [], ""))
            norm_name = self._normalize_school_name(ten_truong.strip())
            quality_score = float(self.quality_scores.get(norm_name, 0.0))

            thanh_pho = str(self._get(row, ['Thành phố', 'tinh_thanh_pho', 'Thanh pho'], ""))
            # Ưu tiên lấy loại trường từ file data_da_hop_nhat.csv đã chuẩn hóa
            loai_raw = self.school_types.get(norm_name, str(self._get(row, ['Loại trường', 'loai_truong'], "Công lập")))
            
            # Chuẩn hóa loại trường thành 3 nhóm chuẩn: Công lập, Tư thục, Quốc tế
            loai = "Công lập"
            loai_lower = loai_raw.lower()
            if any(kw in loai_lower for kw in ["quốc tế", "quoc te"]): 
                loai = "Quốc tế"
            elif any(kw in loai_lower for kw in ["tư thục", "tu thuc", "dân lập", "dan lap", "tư nhân"]):
                loai = "Tư thục"
            elif any(kw in loai_lower for kw in ["công", "cong"]):
                loai = "Công lập"
            
            # Nhận diện vùng miền (Ưu tiên lấy từ file đã chuẩn hóa)
            region_val = self.school_regions.get(norm_name, "Khác")
            if region_val == "Khác":
                text_to_search = self._clean_text(thanh_pho + " " + ten_truong)
                if any(c in text_to_search for c in ['ha noi', 'hai phong', 'bac bo']): 
                    region_val = "Đồng bằng sông Hồng"
                elif any(c in text_to_search for c in ['ho chi minh', 'tp hcm', 'dong nam bo']): 
                    region_val = "Đông Nam Bộ"
                elif any(c in text_to_search for c in ['da nang', 'hue', 'mien trung']): 
                    region_val = "Bắc Trung Bộ"
                elif any(c in text_to_search for c in ['can tho', 'mien tay']): 
                    region_val = "Đồng bằng sông Cửu Long"
            
            # Xác định từ khóa vùng miền chuẩn để filter ở frontend (Tránh trùng lặp Bắc/Bắc Trung Bộ)
            region_key = "Khác"
            if any(m in region_val for m in ["Đồng bằng sông Hồng", "Trung du và miền núi phía Bắc"]):
                region_key = "Bac"
            elif any(m in region_val for m in ["Bắc Trung Bộ", "Duyên hải Nam Trung Bộ và Tây Nguyên"]):
                region_key = "Trung"
            elif any(m in region_val for m in ["Đông Nam Bộ"]):
                region_key = "Nam"
            elif any(m in region_val for m in ["Đồng bằng sông Cửu Long"]):
                region_key = "Tay"

            region_match = 1.0 if preferred_region and preferred_region in region_map and \
                           any((m.lower() in region_val.lower()) for m in region_map[preferred_region]) else 0.0
            
            city_match = 1.0 if preferred_city and preferred_city.lower() in thanh_pho.lower() else 0.0
            type_match = 1.0 if preferred_type and preferred_type.lower() in loai.lower() else 0.0

            recommendations.append({
                "Truong": ten_truong, "Nganh": ten_nganh_raw, "Block": valid_block,
                "Score": round(current_score, 2), "Cutoff": cutoff, "Prob": current_prob,
                "DanhGia": current_danhgia, "Loai": loai, "ThanhPho": thanh_pho, "Region": region_val, "Region_Key": region_key,
                # Fields cho TOPSIS
                "Level_Match": level_match, "Region_Match": region_match, "Type_Match": type_match,
                "City_Match": city_match, "Quality_Score": quality_score, "Major_Score": major_score,
                "Close_Match": close_match
            })

        if not recommendations: return []

        # --- BƯỚC 4: THỰC THI TOPSIS ---
        res_df = pd.DataFrame(recommendations)
        topsis_weights = self._calculate_topsis_weights(priority_order)
        res_df = self._compute_topsis(res_df, topsis_weights)

        # Sắp xếp theo TOPSIS_Score và lấy tập dữ liệu đủ lớn cho bộ lọc (1000 trường)
        top_results = (
            res_df.sort_values(by='TOPSIS_Score', ascending=False)
            .head(1000)
            .to_dict('records')
        )
        return top_results
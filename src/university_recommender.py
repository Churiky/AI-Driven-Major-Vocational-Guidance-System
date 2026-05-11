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
        quality_path = os.path.join(os.path.dirname(path), 'data_da_hop_nhat.csv')
        if os.path.exists(quality_path):
            try:
                df_quality = pd.read_csv(quality_path, encoding='utf-8-sig')
                if 'Tên trường' in df_quality.columns and 'Tổng điểm' in df_quality.columns:
                    # Chuyển đổi sang string và xóa khoảng trắng để map cho chuẩn
                    df_quality['Tên trường'] = df_quality['Tên trường'].astype(str).str.strip()
                    self.quality_scores = dict(zip(df_quality['Tên trường'], df_quality['Tổng điểm']))
            except Exception as e:
                print(f"Không thể tải điểm chất lượng: {e}")

    # -----------------------------------------------------------------------
    # Tiền xử lý
    # -----------------------------------------------------------------------

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

    def recommend(self, nganh_key, user_scores, weights=None):
        """
        Pipeline chính:
          Lọc ngành → Lọc khối truyền thống (regex) → Tính điểm →
          Fallback N/A → Dự đoán xác suất (KNN) → Xếp hạng TOPSIS

        Args:
            nganh_key   : key trong self.mapping, ví dụ "CongNgheThongTin"
            user_scores : dict điểm các môn, ví dụ {"Toan": 8.5, "Ly": 7, "Hoa": 8}
            weights     : [w_diem, w_xacxuat, w_loai, w_vitri, w_chatluong], mặc định [0.35, 0.25, 0.15, 0.1, 0.15]

        Returns:
            list[dict] – Top 10 trường xếp hạng theo TOPSIS, mỗi dict gồm:
                Truong, Nganh, Block, Score, Cutoff, Prob, DanhGia,
                Loai, ThanhPho, Score_TOPSIS
        """
        if weights is None:
            # Tăng thêm 1 trọng số cho điểm chất lượng đào tạo
            weights = [0.35, 0.25, 0.15, 0.1, 0.15]

        keywords = self.mapping.get(nganh_key, [])
        if not keywords:
            return []

        # Tên cột linh hoạt (hỗ trợ cả CSV cũ lẫn CSV mới)
        col_truong  = self._find_col(['truong', 'Tên trường', 'ten_truong'])
        col_nganh   = self._find_col(['ten_nganh', 'Tên ngành', 'Nganh'])
        col_cutoff  = self._find_col(['diem_chuan', 'Diem chuan']) # Đã xóa 'Tổng điểm' khỏi đây
        col_tohop   = self._find_col(['to_hop', 'To hop'])

        if not col_cutoff or not col_tohop:
            raise ValueError("CSV thiếu cột điểm chuẩn hoặc tổ hợp môn.")

        recommendations = []

        for _, row in self.df.iterrows():

            # --- BƯỚC 1: LỌC ĐIỂM CHUẨN ---
            try:
                cutoff = float(row[col_cutoff])
            except (ValueError, TypeError):
                continue

            # Loại bỏ thang điểm ĐGNL (> 35) hoặc bất thường
            if cutoff > 35:
                continue

            # --- BƯỚC 2: LỌC NGÀNH ---
            ten_nganh_raw = str(self._get(row, [col_nganh] if col_nganh else [], ""))
            ten_nganh_clean = self._clean_text(ten_nganh_raw)

            if not any(k in ten_nganh_clean for k in keywords):
                continue

            # --- BƯỚC 3: LỌC KHỐI THI TRUYỀN THỐNG (REGEX) ---
            to_hop_raw = str(row[col_tohop])
            # Hỗ trợ dấu phân cách ";" hoặc ","
            sep = ";" if ";" in to_hop_raw else ","
            blocks = [b.strip() for b in to_hop_raw.split(sep)]

            best_score = -1
            valid_block = ""

            for b in blocks:
                if not b:
                    continue
                # Chỉ lấy khối truyền thống: A00, B01, D07, C03A...
                if not re.match(r'^[A-Z][0-9]{2}[A-Z]?$', b):
                    continue

                score = self.mapper.calculate_score(b, user_scores)
                if score is not None and score > 0:
                    if score > best_score:
                        best_score = score
                        valid_block = b

            # --- BƯỚC 4: FALLBACK N/A (giữ lại để JS lọc "Tất cả") ---
            if best_score == -1:
                current_score = 0
                current_prob = 0
                current_danhgia = "Tỉ lệ thấp"
                # Lấy khối đầu tiên hợp lệ làm đại diện hiển thị
                valid_block = "N/A"
                for b in blocks:
                    if re.match(r'^[A-Z][0-9]{2}[A-Z]?$', b.strip()):
                        valid_block = b.strip()
                        break
            else:
                current_score = best_score
                current_prob = self.predictor.predict(current_score, cutoff)

                diff = current_score - cutoff
                if diff >= 1.0:
                    current_danhgia = "An toàn"
                elif diff >= -0.5:
                    current_danhgia = "Vừa sức"
                else:
                    current_danhgia = "Tỉ lệ thấp"

            ten_truong = str(self._get(row, [col_truong] if col_truong else [], ""))
            ten_truong_clean = ten_truong.strip()
            
            # Lấy điểm chất lượng từ mapping, nếu không có thì lấy mặc định 45.0
            quality_score = self.quality_scores.get(ten_truong_clean, 45.0)

            type_val = float(row.get('Type_Score', 0.5))
            loc_val = float(row.get('Location_Score', 0.5))
            loai = str(self._get(row, ['Loại trường', 'loai_truong'], ""))
            thanh_pho = str(self._get(row, ['Thành phố', 'tinh_thanh_pho'], ""))

            recommendations.append({
                "Truong":    ten_truong,
                "Nganh":     ten_nganh_raw,
                "Block":     valid_block,
                "Score":     round(current_score, 2),
                "Cutoff":    cutoff,
                "Prob":      current_prob,
                "DanhGia":   current_danhgia,
                "Loai":      loai,
                "ThanhPho":  thanh_pho,
                "Type_Val":  type_val,
                "Loc_Val":   loc_val,
                "Quality_Score": float(quality_score)
            })

        if not recommendations:
            return []

        # --- BƯỚC 5: XẾP HẠNG ĐA TIÊU CHÍ (TOPSIS) ---
        res_df = pd.DataFrame(recommendations)

        # Ma trận quyết định: C1(Sát điểm), C2(Xác suất), C3(Loại trường), C4(Vị trí), C5(Chất lượng)
        matrix = []
        for _, r in res_df.iterrows():
            c1 = r['Cutoff'] # Càng sát điểm chuẩn càng tốt
            c2 = r['Prob']
            c3 = r['Type_Val']
            c4 = r['Loc_Val']
            c5 = r['Quality_Score'] # Điểm chất lượng trường
            matrix.append([c1, c2, c3, c4, c5])

        matrix = np.array(matrix, dtype=float)

        # 1. Chuẩn hóa vector
        norm = np.sqrt((matrix ** 2).sum(axis=0) + 1e-9)
        norm_matrix = matrix / norm

        # 2. Nhân trọng số
        weighted = norm_matrix * np.array(weights)

        # 3. PIS (+) và NIS (-)
        pis = np.max(weighted, axis=0)
        nis = np.min(weighted, axis=0)

        # 4. Khoảng cách Euclidean
        d_plus  = np.sqrt(((weighted - pis) ** 2).sum(axis=1))
        d_minus = np.sqrt(((weighted - nis) ** 2).sum(axis=1))

        # 5. Điểm tương đồng TOPSIS
        res_df['Score_TOPSIS'] = d_minus / (d_plus + d_minus + 1e-9)

        # Trả về Top 10, bỏ cột nội bộ
        top10 = (
            res_df
            .sort_values(by='Score_TOPSIS', ascending=False)
            .head(10)
            .drop(columns=['Type_Val', 'Loc_Val'], errors='ignore')
            .to_dict('records')
        )
        return top10
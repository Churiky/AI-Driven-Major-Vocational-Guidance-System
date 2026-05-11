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

    def recommend(self, nganh_key, user_scores, weights=None, preferred_region=None, preferred_city=None, preferred_type=None, priority_order=None, expand_majors=False):
        """
        Pipeline chính hỗ trợ ưu tiên động: Theo thứ tự người dùng kéo thả.
        """
        # Bảng ánh xạ vùng miền chuẩn theo dữ liệu CSV thực tế
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
            # --- BƯỚC 1: LỌC ĐIỂM CHUẨN ---
            try:
                cutoff = float(row[col_cutoff])
            except (ValueError, TypeError): continue
            if cutoff > 35: continue

            # --- BƯỚC 2: LỌC NGÀNH ---
            ten_nganh_raw = str(self._get(row, [col_nganh] if col_nganh else [], ""))
            ten_nganh_clean = self._clean_text(ten_nganh_raw)
            
            major_score = 0
            # Nếu không ở chế độ mở rộng, lọc khắt khe theo keywords AI
            if not expand_majors:
                if not any(k in ten_nganh_clean for k in keywords): 
                    continue
                # Tính độ ưu tiên ngành: nếu tên ngành chứa chính xác từ khóa chính thì điểm cao
                major_score = 100 if any(k == ten_nganh_clean for k in keywords) else 50
            else:
                # Chế độ mở rộng: ưu tiên ngành đúng nhưng không loại bỏ ngành khác
                major_score = 100 if any(k in ten_nganh_clean for k in keywords) else 0

            # --- BƯỚC 3: TÍNH ĐIỂM ---
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

            diff = current_score - cutoff
            if diff >= 1.5: current_danhgia = "An toàn"
            elif diff >= -1.5: current_danhgia = "Vừa sức"
            else: current_danhgia = "Tỉ lệ thấp"
            
            # Gắn nhãn bổ sung: Các trường cực kỳ sát điểm (±1.5)
            close_match = 1 if abs(diff) <= 1.5 else 0

            ten_truong = str(self._get(row, [col_truong] if col_truong else [], ""))
            norm_name = self._normalize_school_name(ten_truong.strip())
            quality_score = self.quality_scores.get(norm_name, 0.0)

            # --- XỬ LÝ VÙNG MIỀN (NHẬN DIỆN THÔNG MINH HƠN) ---
            thanh_pho = str(self._get(row, ['Thành phố', 'tinh_thanh_pho', 'Thanh pho'], ""))
            loai = str(self._get(row, ['Loại trường', 'loai_truong'], ""))
            ten_truong = str(self._get(row, [col_truong] if col_truong else [], ""))
            
            region_val = "Khác"
            # Gộp cả Tỉnh/Thành phố và Tên trường để tìm kiếm từ khóa
            text_to_search = self._clean_text(thanh_pho + " " + ten_truong)
            
            # Danh sách từ khóa mở rộng cho từng vùng miền
            north_keywords = [
                'ha noi', 'hai phong', 'thai nguyen', 'nam dinh', 'bac ninh', 'vinh phuc', 'phu tho', 
                'quang ninh', 'hai duong', 'hung yen', 'bac giang', 'ha nam', 'ninh binh', 'thai binh',
                'hoa binh', 'son la', 'dien bien', 'lai chau', 'lao cai', 'yen bai', 'tuyen quang', 
                'lang son', 'cao bang', 'ha giang', 'bac kan', 'bac bo', 'tay bac', 'dong bac',
                'hn', 'hp', 'tn', 'nd', 'bn', 'vp', 'pt', 'qn', 'hd', 'hy', 'bg', 'hn', 'nb', 'tb'
            ]
            central_keywords = [
                'da nang', 'hue', 'nghe an', 'thanh hoa', 'nha trang', 'binh dinh', 'quang nam', 
                'quang ngai', 'quang tri', 'quang binh', 'ha tinh', 'phu yen', 'ninh thuan', 'binh thuan',
                'kon tum', 'gia lai', 'dak lak', 'dak nong', 'lam dong', 'da lat', 'tay nguyen', 'mien trung'
            ]
            south_keywords = [
                'ho chi minh', 'tp hcm', 'tphcm', 'binh duong', 'dong nai', 'vung tau', 'ba ria', 
                'tay ninh', 'binh phuoc', 'dong nam bo', 'mien nam'
            ]
            west_keywords = [
                'can tho', 'an giang', 'kien giang', 'vinh long', 'ben tre', 'tra vinh', 'long an', 
                'dong thap', 'hau giang', 'soc trang', 'bac lieu', 'ca mau', 'tien giang', 
                'dong bang song cuu long', 'mien tay'
            ]

            if any(c in text_to_search for c in north_keywords):
                # Ưu tiên map về 2 vùng lớn ở miền Bắc
                if any(c in text_to_search for c in ['thai nguyen', 'phu tho', 'son la', 'hoa binh', 'tuyen quang', 'lang son', 'tay bac', 'dong bac']):
                    region_val = "Trung du và miền núi phía Bắc"
                else:
                    region_val = "Đồng bằng sông Hồng"
            elif any(c in text_to_search for c in south_keywords):
                region_val = "Đông Nam Bộ"
            elif any(c in text_to_search for c in central_keywords):
                if any(c in text_to_search for c in ['kon tum', 'gia lai', 'dak lak', 'dak nong', 'lam dong', 'da lat', 'tay nguyen']):
                    region_val = "Duyên hải Nam Trung Bộ và Tây Nguyên"
                else:
                    region_val = "Bắc Trung Bộ"
            elif any(c in text_to_search for c in west_keywords):
                region_val = "Đồng bằng sông Cửu Long"
            
            # Nếu vẫn là "Khác", mới lấy từ file chất lượng (mapping theo tên trường)
            if region_val == "Khác":
                region_val = self.school_regions.get(norm_name, "Khác")

            # 4.1 Khớp vùng miền (So sánh với ưu tiên của người dùng)
            region_match = 0
            if preferred_region and preferred_region in region_map:
                target_regions = region_map[preferred_region]
                # Nếu region_val khớp với mục tiêu hoặc ngược lại
                if any((m.lower() in region_val.lower() or region_val.lower() in m.lower()) for m in target_regions):
                    region_match = 1
            
            city_match   = 1 if preferred_city and preferred_city.lower() in thanh_pho.lower() else 0
            type_match   = 1 if preferred_type and preferred_type.lower() in loai.lower() else 0
            # Level_Match = 1 nếu là An toàn hoặc Vừa sức (người dùng có khả năng đỗ)
            level_match  = 1 if current_danhgia in ["An toàn", "Vừa sức"] else 0

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
                "Quality_Score": float(quality_score),
                "Region":    region_val,
                "Region_Match": region_match,
                "City_Match":   city_match,
                "Type_Match":   type_match,
                "Level_Match":  level_match,
                "Close_Match":  close_match,
                "Major_Score":  major_score
            })

        if not recommendations: return []

        # --- BƯỚC 5: XẾP HẠNG ĐỘNG ---
        res_df = pd.DataFrame(recommendations)
        # Mặc định: Chất lượng > Vùng miền > Loại trường > Tỉnh thành
        sort_keys = ['Level_Match', 'Region_Match', 'Type_Match', 'City_Match']
        
        if priority_order:
            key_map = {
                'chat_luong': 'Level_Match',
                'vung_mien': 'Region_Match',
                'loai_truong': 'Type_Match',
                'tinh_thanh': 'City_Match'
            }
            sort_keys = [key_map.get(k.strip()) for k in priority_order.split(',') if k.strip() in key_map]
            
            # Close_Match và Major_Score sẽ là các tiêu chí phụ cuối cùng
            if 'Major_Score' not in sort_keys:
                sort_keys.append('Major_Score')
            if 'Close_Match' not in sort_keys:
                sort_keys.append('Close_Match')

        # Luôn thêm Điểm chuẩn và Chất lượng vào cuối để so sánh nếu các tiêu chí trên bằng nhau
        sort_keys += ['Cutoff', 'Quality_Score']

        top10 = (
            res_df
            .sort_values(by=sort_keys, ascending=[False] * len(sort_keys))
            .head(2000)
            .to_dict('records')
        )
        return top10
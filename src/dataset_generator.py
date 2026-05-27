import pandas as pd
import random
import numpy as np

class DatasetGenerator:
    def round_score(self, score):
        """Làm tròn điểm đến 0.25 theo quy tắc thi THPT"""
        return round(score * 4) / 4

    def generate(self, n_samples=10000): # Tăng lên 10k mẫu để học nhiều ngành
        data = []

        # Định nghĩa các tổ hợp ngành và đặc điểm nhận diện
        # Format: (Tên ngành, Hệ thi ưu tiên, Môn trọng tâm, Holland Key, Mã ngành, Đặc trưng riêng)
        major_rules = [
            # KHTN - majors with clearer separation
            ("CongNgheThongTin", "KHTN", ["Toan", "Ly"], ["R", "I"], "CNTT", {"ly_hoa_diff": 1.0, "toan_ly_ratio": 1.2}),
            ("YKhoa", "KHTN", ["Hoa", "Sinh"], ["I", "S"], "YK", {"sinh_hoa_diff": 0.8, "toa_min": 5.5}),
            ("KỹThuatOTo", "KHTN", ["Toan", "Ly"], ["R", "C"], "KTO", {"toan_ly_diff": 1.5, "toa_max": 8.0}),
            ("KienTruc", "KHTN", ["Toan", "Van"], ["A", "R"], "KT", {"toan_van_balance": 0.3, "van_min": 6.0}),
            ("CongNgheSinhHoc", "KHTN", ["Hoa", "Sinh"], ["I", "R"], "CNSH", {"sinh_hoa_ratio": 1.3, "toa_max": 7.5}),
            ("CongNgheVatLieu", "KHTN", ["Ly", "Hoa"], ["I", "R"], "CNVL", {"ly_hoa_balance": 0.2, "toa_min": 5.0}),
            ("DienTuVienThong", "KHTN", ["Toan", "Ly"], ["R", "I"], "DTVT", {"toan_ly_diff": 1.2, "toa_max": 8.5}),
            ("CongNghiepBaoVeMoiTruong", "KHTN", ["Ly", "Hoa"], ["I", "S"], "CNBMT", {"ly_hoa_diff": 1.0, "toa_min": 5.5}),

            # KHXH - majors with clearer separation
            ("KinhTe", "KHXH", ["Toan", "Anh"], ["E", "C"], "KT", {"toan_anh_diff": 0.8, "toa_min": 5.5}),
            ("Marketing", "KHXH", ["Anh", "Van"], ["E", "A"], "MKT", {"anh_van_diff": 1.0, "van_max": 8.5}),
            ("Luat", "KHXH", ["Van", "Su"], ["C", "S"], "LUAT", {"van_su_balance": 0.3, "su_min": 5.5}),
            ("NgonNguAnh", "KHXH", ["Anh", "Van"], ["S", "A"], "NNA", {"anh_van_ratio": 1.4, "toa_max": 8.0}),
            ("TamLyHoc", "KHXH", ["Van", "Anh"], ["S", "I"], "TLH", {"van_anh_diff": 0.9, "toa_min": 5.0}),
            ("SuPham", "KHXH", ["Van", "Su"], ["S", "C"], "SP", {"van_su_diff": 1.2, "toa_max": 8.0}),
            ("QuanTriKhachSan", "KHXH", ["Anh", "Van"], ["E", "S"], "QTKS", {"anh_van_balance": 0.2, "toa_min": 5.5}),
            ("DuLich", "KHXH", ["Van", "Anh"], ["E", "S"], "DL", {"anh_van_diff": 1.1, "toa_max": 8.5}),
            ("QuanTriTaiChinh", "KHXH", ["Toan", "Anh"], ["E", "C"], "QTTC", {"toan_anh_diff": 1.0, "toa_min": 5.0}),
            ("KinhTeQuocTe", "KHXH", ["Toan", "Anh"], ["E", "I"], "KQT", {"toan_anh_ratio": 1.3, "toa_max": 8.0}),
            ("NgoaiGiao", "KHXH", ["Van", "Su"], ["E", "S"], "NG", {"van_su_diff": 1.0, "toa_min": 5.5}),
            ("KeToan", "KHXH", ["Toan", "Anh"], ["C", "E"], "KTOAN", {"toan_anh_diff": 1.5, "toa_max": 8.5}),  # Increased separation
            ("DuocLieu", "KHXH", ["Hoa", "Sinh"], ["I", "C"], "DLIEU", {"sinh_hoa_diff": 1.0, "toa_min": 5.0})
        ]

        for _ in range(n_samples):
            # Chọn ngẫu nhiên 1 quy tắc ngành để tạo dữ liệu "chuẩn" cho ngành đó
            target_major, system, core_subjects, holland_keys, major_code, special_traits = random.choice(major_rules)

            # Khởi tạo điểm mặc định trung bình
            scores = {
                "Toan": self.round_score(random.uniform(4, 7)),
                "Van": self.round_score(random.uniform(4, 7)),
                "Anh": self.round_score(random.uniform(4, 7)),
                "Ly": 0, "Hoa": 0, "Sinh": 0, "Su": 0, "Dia": 0, "GDCD": 0
            }

            if system == "KHTN":
                scores["Ly"] = self.round_score(random.uniform(4, 7))
                scores["Hoa"] = self.round_score(random.uniform(4, 7))
                scores["Sinh"] = self.round_score(random.uniform(4, 7))
            else:
                scores["Su"] = self.round_score(random.uniform(4, 7))
                scores["Dia"] = self.round_score(random.uniform(4, 7))
                scores["GDCD"] = self.round_score(random.uniform(4, 7))

            # TĂNG ĐIỂM môn trọng tâm cho ngành target với sự phân biệt mạnh hơn
            for sub in core_subjects:
                base_score = random.uniform(7.5, 10)
                # Thêm variance dựa trên đặc trưng ngành
                if "toan_ly_diff" in special_traits and sub in ["Toan", "Ly"]:
                    # Tạo sự chênh lệch giữa Toan và Lý
                    if sub == "Toan":
                        scores[sub] = self.round_score(base_score + special_traits["toan_ly_diff"] * random.uniform(-0.5, 0.5))
                    else:
                        scores[sub] = self.round_score(base_score - special_traits["toan_ly_diff"] * random.uniform(-0.5, 0.5))
                elif "ly_hoa_diff" in special_traits and sub in ["Ly", "Hoa"]:
                    # Tạo sự chênh lệch giữa Lý và Hóa
                    if sub == "Ly":
                        scores[sub] = self.round_score(base_score + special_traits["ly_hoa_diff"] * random.uniform(-0.5, 0.5))
                    else:
                        scores[sub] = self.round_score(base_score - special_traits["ly_hoa_diff"] * random.uniform(-0.5, 0.5))
                elif "van_anh_diff" in special_traits and sub in ["Van", "Anh"]:
                    # Tạo sự chênh lệch giữa Văn và Anh
                    if sub == "Van":
                        scores[sub] = self.round_score(base_score + special_traits["van_anh_diff"] * random.uniform(-0.5, 0.5))
                    else:
                        scores[sub] = self.round_score(base_score - special_traits["van_anh_diff"] * random.uniform(-0.5, 0.5))
                elif "van_su_diff" in special_traits and sub in ["Van", "Su"]:
                    # Tạo sự chênh lệch giữa Văn và Sử
                    if sub == "Van":
                        scores[sub] = self.round_score(base_score + special_traits["van_su_diff"] * random.uniform(-0.5, 0.5))
                    else:
                        scores[sub] = self.round_score(base_score - special_traits["van_su_diff"] * random.uniform(-0.5, 0.5))
                elif "sinh_hoa_diff" in special_traits and sub in ["Sinh", "Hoa"]:
                    # Tạo sự chênh lệch giữa Sinh và Hóa
                    if sub == "Sinh":
                        scores[sub] = self.round_score(base_score + special_traits["sinh_hoa_diff"] * random.uniform(-0.5, 0.5))
                    else:
                        scores[sub] = self.round_score(base_score - special_traits["sinh_hoa_diff"] * random.uniform(-0.5, 0.5))
                else:
                    scores[sub] = self.round_score(base_score)

            # Áp dụng các constraint đặc biệt
            if "toa_min" in special_traits:
                min_score = special_traits["toa_min"]
                # Đảm bảo một trong các môn tối thiểu đạt được
                if system == "KHTN":
                    if "Toan" in scores and scores["Toan"] < min_score:
                        scores["Toan"] = self.round_score(min_score)
                    if "Ly" in scores and scores["Ly"] < min_score:
                        scores["Ly"] = self.round_score(min_score)
                else:
                    if "Toan" in scores and scores["Toan"] < min_score:
                        scores["Toan"] = self.round_score(min_score)
                    if "Van" in scores and scores["Van"] < min_score:
                        scores["Van"] = self.round_score(min_score)

            if "toa_max" in special_traits:
                max_score = special_traits["toa_max"]
                # Giới hạn điểm tối đa để tránh quáDegree
                if system == "KHTN":
                    if "Toan" in scores and scores["Toan"] > max_score:
                        scores["Toan"] = self.round_score(max_score)
                    if "Ly" in scores and scores["Ly"] > max_score:
                        scores["Ly"] = self.round_score(max_score)
                else:
                    if "Toan" in scores and scores["Toan"] > max_score:
                        scores["Toan"] = self.round_score(max_score)
                    if "Van" in scores and scores["Van"] > max_score:
                        scores["Van"] = self.round_score(max_score)

            # TẠO ĐIỂM HOLLAND (1-5) với sự phân biệt rõ ràng hơn
            h = {k: random.randint(1, 3) for k in ["R", "I", "A", "S", "E", "C"]}
            # Tăng điểm Holland tương ứng với ngành + thêm noise
            for key in holland_keys:
                base_boost = random.randint(4, 5)
                # Thêm sự ngẫu nhiên để tránh pattern quá rigid
                h[key] = min(5, base_boost + random.randint(-1, 1))

            # Tính điểm trung bình tổ hợp
            khtn_avg = round((scores["Ly"] + scores["Hoa"] + scores["Sinh"]) / 3, 2) if system == "KHTN" else 0
            khxh_avg = round((scores["Su"] + scores["Dia"] + scores["GDCD"]) / 3, 2) if system == "KHXH" else 0

            # THÊM FEATURE INTERACTIONS để giúp mô hình học pattern phức tạp
            # Đây là các đặc trunks mới sẽ được tính toán từ scores hiện có
            # Nhưng để tetap tương thích với 17 features cũ, chúng ta sẽ lưu trữ chúng tạm thời
            # và sau này có thể mở rộng feature set trong transformer

            # Temporarily store interaction features (will be used if we expand features later)
            interaction_features = {
                "toan_ly": scores["Toan"] * scores["Ly"] if system == "KHTN" else 0,
                "van_su": scores["Van"] * scores["Su"] if system == "KHXH" else 0,
                "toa_anh": scores["Toan"] * scores["Anh"],
                "ly_hoa": scores["Ly"] * scores["Hoa"] if system == "KHTN" else 0,
                "sinh_hoa": scores["Sinh"] * scores["Hoa"] if system == "KHTN" else 0,
                "van_anh": scores["Van"] * scores["Anh"] if system == "KHXH" else 0
            }

            data.append([
                scores["Toan"], scores["Van"], scores["Anh"],
                scores["Ly"], scores["Hoa"], scores["Sinh"],
                scores["Su"], scores["Dia"], scores["GDCD"],
                khtn_avg, khxh_avg,
                h["R"], h["I"], h["A"], h["S"], h["E"], h["C"],
                target_major
            ])

        df = pd.DataFrame(data, columns=[
            "Toan", "Van", "Anh", "Ly", "Hoa", "Sinh", "Su", "Dia", "GDCD",
            "KHTN", "KHXH", "R", "I", "A", "S", "E", "C", "Major"
        ])

        df.to_csv("data/synthetic_students.csv", index=False)
        print(f"Generated Dataset with {len(major_rules)} majors and {len(df)} samples.")

if __name__ == "__main__":
    DatasetGenerator().generate()
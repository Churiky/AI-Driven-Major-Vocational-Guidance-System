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
        # Format: (Tên ngành, Hệ thi ưu tiên, Môn trọng tâm, Holland Key)
        major_rules = [
            ("CongNgheThongTin", "KHTN", ["Toan", "Ly"], ["R", "I"]),
            ("YKhoa", "KHTN", ["Hoa", "Sinh"], ["I", "S"]),
            ("KỹThuatOTo", "KHTN", ["Toan", "Ly"], ["R", "C"]),
            ("KienTruc", "KHTN", ["Toan", "Van"], ["A", "R"]),
            ("KinhTe", "KHXH", ["Toan", "Anh"], ["E", "C"]),
            ("Marketing", "KHXH", ["Anh", "Van"], ["E", "A"]),
            ("Luat", "KHXH", ["Van", "Su"], ["C", "S"]),
            ("NgonNguAnh", "KHXH", ["Anh", "Van"], ["S", "A"]),
            ("TamLyHoc", "KHXH", ["Van", "Anh"], ["S", "I"]),
            ("SuPham", "KHXH", ["Van", "Su"], ["S", "C"]),
            ("QuanTriKhachSan", "KHXH", ["Anh", "Van"], ["E", "S"]),
            ("KeToan", "KHXH", ["Toan", "Anh"], ["C", "E"])
        ]

        for _ in range(n_samples):
            # Chọn ngẫu nhiên 1 quy tắc ngành để tạo dữ liệu "chuẩn" cho ngành đó
            target_major, system, core_subjects, holland_keys = random.choice(major_rules)

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

            # TĂNG ĐIỂM môn trọng tâm cho ngành target
            for sub in core_subjects:
                scores[sub] = self.round_score(random.uniform(7.5, 10))

            # TẠO ĐIỂM HOLLAND (1-5)
            h = {k: random.randint(1, 3) for k in ["R", "I", "A", "S", "E", "C"]}
            # Tăng điểm Holland tương ứng với ngành
            for key in holland_keys:
                h[key] = random.randint(4, 5)

            # Tính điểm trung bình tổ hợp
            khtn_avg = round((scores["Ly"] + scores["Hoa"] + scores["Sinh"]) / 3, 2) if system == "KHTN" else 0
            khxh_avg = round((scores["Su"] + scores["Dia"] + scores["GDCD"]) / 3, 2) if system == "KHXH" else 0

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
        print(f"✅ Đã tạo Dataset với {len(major_rules)} ngành và {len(df)} mẫu.")

if __name__ == "__main__":
    DatasetGenerator().generate()
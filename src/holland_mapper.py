class HollandMapper:

    def __init__(self):

        self.mapping = {
            "R": ["CoKhi","Dien","XayDung"], #thích làm việc thực hành, kỹ thuật
            "I": ["CongNgheThongTin","TriTueNhanTao","KhoaHocDuLieu"], # thích nghiên cứu, phân tích
            "A": ["ThietKe","TruyenThong"], # sáng tạo, nghệ thuật
            "S": ["YKhoa","GiaoDuc"], # thích giúp đỡ người khác
            "E": ["KinhDoanh","Marketing"], # lãnh đạo, kinh doanh
            "C": ["KeToan","TaiChinh"] # làm việc có tổ chức, dữ liệu
        }

    def suggest_major(self, holland_scores):

        top = max(holland_scores, key=holland_scores.get)

        return self.mapping[top]
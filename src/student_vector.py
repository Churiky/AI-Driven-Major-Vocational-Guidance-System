class StudentVectorBuilder:
    def build(self, scores, holland):
        # Đã chuẩn hóa thành 17 features và đồng nhất thứ tự với app.py
        vector = [
            scores.get("toan", 0),
            scores.get("van", 0),
            scores.get("anh", 0),
            scores.get("ly", 0),
            scores.get("hoa", 0),
            scores.get("sinh", 0),
            scores.get("su", 0),
            scores.get("dia", 0),
            scores.get("gdcd", 0),
            scores.get("khtn", 0),
            scores.get("khxh", 0),
            holland.get("R", 0),
            holland.get("I", 0),
            holland.get("A", 0),
            holland.get("S", 0),
            holland.get("E", 0),
            holland.get("C", 0)
        ]
        return vector
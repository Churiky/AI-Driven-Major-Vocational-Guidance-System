class SubjectMapper:

    def __init__(self):

        self.mapping = {

            # ===== A =====
            "A00": ["toan","ly","hoa"],
            "A01": ["toan","ly","anh"],
            "A02": ["toan","ly","sinh"],
            "A03": ["toan","ly","su"],
            "A04": ["toan","ly","dia"],
            "A05": ["toan","hoa","su"],
            "A06": ["toan","hoa","dia"],
            "A07": ["toan","su","dia"],
            "A08": ["toan","su","gdcd"],
            "A09": ["toan","dia","gdcd"],
            "A10": ["toan","ly","gdcd"],
            "A11": ["toan","hoa","gdcd"],
            "A12": ["toan","khtn","khxh"],
            "A14": ["toan","khtn","dia"],
            "A15": ["toan","khtn","gdcd"],
            "A16": ["toan","khtn","van"],
            "A17": ["toan","ly","khxh"],
            "A18": ["toan","hoa","khxh"],

            # ===== B =====
            "B00": ["toan","hoa","sinh"],
            "B01": ["toan","sinh","su"],
            "B02": ["toan","sinh","dia"],
            "B03": ["toan","sinh","van"],
            "B04": ["toan","sinh","gdcd"],
            "B05": ["toan","sinh","khxh"],
            "B08": ["toan","sinh","anh"],

            # ===== C =====
            "C00": ["van","su","dia"],
            "C01": ["van","toan","ly"],
            "C02": ["van","toan","hoa"],
            "C03": ["van","toan","su"],
            "C04": ["van","toan","dia"],
            "C05": ["van","ly","hoa"],
            "C06": ["van","ly","sinh"],
            "C07": ["van","ly","su"],
            "C08": ["van","hoa","sinh"],
            "C09": ["van","ly","dia"],
            "C10": ["van","hoa","su"],
            "C12": ["van","sinh","su"],
            "C13": ["van","sinh","dia"],
            "C14": ["van","toan","gdcd"],
            "C15": ["van","toan","khxh"],
            "C16": ["van","ly","gdcd"],
            "C17": ["van","hoa","gdcd"],
            "C19": ["van","su","gdcd"],
            "C20": ["van","dia","gdcd"],

            # ===== D =====
            "D01": ["van","toan","anh"],
            "D02": ["van","toan","ngoai_ngu"],
            "D03": ["van","toan","ngoai_ngu"],
            "D04": ["van","toan","ngoai_ngu"],
            "D05": ["van","toan","ngoai_ngu"],
            "D06": ["van","toan","ngoai_ngu"],

            "D07": ["toan","hoa","anh"],
            "D08": ["toan","sinh","anh"],
            "D09": ["toan","su","anh"],
            "D10": ["toan","dia","anh"],

            "D11": ["van","ly","anh"],
            "D12": ["van","hoa","anh"],
            "D13": ["van","sinh","anh"],
            "D14": ["van","su","anh"],
            "D15": ["van","dia","anh"],

            # nhóm D có ngoại ngữ khác → gom chung
            "D16": ["toan","dia","ngoai_ngu"],
            "D17": ["toan","dia","ngoai_ngu"],
            "D18": ["toan","dia","ngoai_ngu"],
            "D19": ["toan","dia","ngoai_ngu"],
            "D20": ["toan","dia","ngoai_ngu"],

            "D21": ["toan","hoa","ngoai_ngu"],
            "D22": ["toan","hoa","ngoai_ngu"],
            "D23": ["toan","hoa","ngoai_ngu"],
            "D24": ["toan","hoa","ngoai_ngu"],
            "D25": ["toan","hoa","ngoai_ngu"],

            "D26": ["toan","ly","ngoai_ngu"],
            "D27": ["toan","ly","ngoai_ngu"],
            "D28": ["toan","ly","ngoai_ngu"],
            "D29": ["toan","ly","ngoai_ngu"],
            "D30": ["toan","ly","ngoai_ngu"],

            "D31": ["toan","sinh","ngoai_ngu"],
            "D32": ["toan","sinh","ngoai_ngu"],
            "D33": ["toan","sinh","ngoai_ngu"],
            "D34": ["toan","sinh","ngoai_ngu"],
            "D35": ["toan","sinh","ngoai_ngu"],

            "D66": ["van","gdcd","anh"],
            "D72": ["van","khtn","anh"],
            "D78": ["van","khxh","anh"],

            "D84": ["toan","gdcd","anh"],
            "D90": ["toan","khtn","anh"],
            "D96": ["toan","khxh","anh"]
        }

    def calculate_score(self, block, scores):

        if block not in self.mapping:
            return None

        subjects = self.mapping[block]  

        total = 0

        for sub in subjects:
            total += scores.get(sub, 0)

        return total
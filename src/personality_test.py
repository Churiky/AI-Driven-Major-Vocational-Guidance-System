class HollandTest:

    def __init__(self):

        self.questions = [
            ("Tôi thích sửa chữa máy móc", "R"),
            ("Tôi thích làm thí nghiệm khoa học", "I"),
            ("Tôi thích vẽ hoặc thiết kế", "A"),
            ("Tôi thích giúp đỡ người khác", "S"),
            ("Tôi thích lãnh đạo nhóm", "E"),
            ("Tôi thích làm việc với dữ liệu", "C")
        ]

        self.scores = {"R":0,"I":0,"A":0,"S":0,"E":0,"C":0}

    def run(self):

        print("\n===== HOLLAND TEST =====")

        for q, t in self.questions:

            score = int(input(f"{q} (1-5): "))

            self.scores[t] += score

        return self.scores
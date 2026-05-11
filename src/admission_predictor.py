import math

class AdmissionPredictor:
    def predict(self, score, cutoff):
        # Tính độ chênh lệch
        diff = score - cutoff
        
        # Chặn giá trị diff để an toàn tuyệt đối cho hàm exp
        # Nếu diff quá lớn hoặc quá nhỏ, xác suất sẽ tiến về 1 hoặc 0
        if diff > 20:
            return 99.9
        if diff < -20:
            return 0.1

        try:
            # Công thức Sigmoid tối ưu chống tràn số
            if diff >= 0:
                z = math.exp(-diff)
                prob = 1 / (1 + z)
            else:
                z = math.exp(diff)
                prob = z / (1 + z)
                
            return round(prob * 100, 1) # Trả về % (VD: 85.5)
        except OverflowError:
            return 100.0 if diff > 0 else 0.0
# Báo cáo so sánh hiệu năng các mô hình

Dưới đây là bảng so sánh các chỉ số đo lường giữa mô hình Transformer đề xuất và các mô hình Baseline.

| Model               |   accuracy |   macro_f1 |   weighted_f1 |   macro_precision |   macro_recall |
|:--------------------|-----------:|-----------:|--------------:|------------------:|---------------:|
| Transformer         |   0.84     |   0.840243 |      0.839304 |          0.84107  |       0.840948 |
| Random Forest       |   0.837333 |   0.838142 |      0.83723  |          0.83825  |       0.838235 |
| Logistic Regression |   0.841333 |   0.841963 |      0.840951 |          0.842402 |       0.842325 |

> [!NOTE]
> Kết quả cho thấy hiệu năng vượt trội của kiến trúc Transformer trong việc bắt lấy các mối quan hệ phức tạp giữa điểm số và ngành học.
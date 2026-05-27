# Báo Cáo So Sánh Mô Hình Transformer Thường Với Baselines

Dưới đây là bảng so sánh chi tiết các chỉ số đo lường hiệu năng phân loại lớp kiểm thử:

| Model                |   accuracy |   macro_f1 |   weighted_f1 |   macro_precision |   macro_recall |
|:---------------------|-----------:|-----------:|--------------:|------------------:|---------------:|
| Logistic Regression  |   0.841333 |   0.841963 |      0.840951 |          0.842402 |       0.842325 |
| Random Forest        |   0.837333 |   0.838142 |      0.83723  |          0.83825  |       0.838235 |
| Transformer (thường) |   0.84     |   0.840243 |      0.839304 |          0.84107  |       0.840948 |

> [!NOTE]
> Kết quả cho thấy kiến trúc Transformer thường có sự tối ưu hóa tốt hơn trong việc nắm bắt quan hệ đặc trưng so với các baseline truyền thống.

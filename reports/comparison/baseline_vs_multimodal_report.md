# Báo Cáo So Sánh Mô Hình Multimodal Transformer Với Baselines

Dưới đây là bảng so sánh chi tiết các chỉ số đo lường hiệu năng phân loại lớp kiểm thử:

| Model                  |   accuracy |   macro_f1 |   weighted_f1 |   macro_precision |   macro_recall |
|:-----------------------|-----------:|-----------:|--------------:|------------------:|---------------:|
| Logistic Regression    |   0.841333 |   0.841963 |      0.840951 |          0.842402 |       0.842325 |
| Random Forest          |   0.837333 |   0.838142 |      0.83723  |          0.83825  |       0.838235 |
| Multimodal Transformer |   0.906667 |   0.904929 |      0.905336 |          0.908912 |       0.905947 |

> [!NOTE]
> Mô hình cải tiến Multimodal Transformer đạt được sự cải thiện đồng đều ở mọi chỉ số phân loại, khẳng định tính hiệu quả của mạng Cross-Attention.

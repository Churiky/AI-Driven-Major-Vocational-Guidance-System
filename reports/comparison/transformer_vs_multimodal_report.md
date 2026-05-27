# Báo Cáo So Sánh - Transformer thường vs Multimodal Transformer

Dưới đây là bảng so sánh chi tiết các chỉ số đo lường hiệu năng phân loại lớp kiểm thử:

| Model                  |   accuracy |   macro_f1 |   weighted_f1 |   macro_precision |   macro_recall |
|:-----------------------|-----------:|-----------:|--------------:|------------------:|---------------:|
| Transformer (thường)   |   0.84     |   0.840243 |      0.839304 |          0.84107  |       0.840948 |
| Multimodal Transformer |   0.906667 |   0.904929 |      0.905336 |          0.908912 |       0.905947 |

> [!NOTE]
> Việc tích hợp thêm luồng thông tin tính cách Holland RIASEC qua cơ chế Cross-Attention giúp mô hình nâng cao đáng kể hiệu năng dự báo chuyên sâu.

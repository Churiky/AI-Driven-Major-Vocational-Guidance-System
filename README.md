# Hỗ Trợ Học Sinh Lớp 12

## Giới thiệu
Dự án Hỗ Trợ Học Sinh Lớp 12 là ứng dụng web giúp học sinh Việt Nam xác định ngành học phù hợp dựa trên bài test Holland (RIASEC), điểm số học môn và sở thích (vùng, thành phố, loại trường). Kết hợp mô hình AI multimodal transformer và thuật toán TOPSIS để xếp hạng trường học phù hợp nhất.

## Yêu cầu hệ thống
- Python 3.9 hoặc cao hơn
- Các gói Python được liệt kê trong `requirements.txt`
- Thiết bị có kết nối internet để truy cập Google Gemini API (nếu sử dụng chatbot)

## Cài đặt
1. Sao chép hoặc tải về mã nguồn.
2. Tiến hành tạo môi trường ảo (tùy chọn nhưng được khuyến nghị):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Cài đặt các gói phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

## Cấu hình
### API Key Gemini
- Tạo file `.env` trong thư mục gốc dự án (đã có trong `.gitignore`).
- Thêm dòng:
  ```
  GEMINI_API_KEY=your_actual_gemini_api_key_here
  ```
- Không được lưu trữ key mã nguồn hoặc commit lên repository.

### Dữ liệu và mô hình
- Đặt file mô hình PyTorch (`*.pth`), scaler và danh sách nhãn (`*.pkl`) vào thư mục `models/`.
- Đặt file dữ liệu điểm chuẩn `diem_chuan_all.csv` và file chất lượng đào tạo `data_da_hop_nhat.csv` vào thư mục `data/`.
- Đặt tài liệu PDF cho chatbot RAG vào thưiekt `data/documents/`.

## Chạy ứng dụng
1. Đảm bảo biến môi trường được tải (nếu sử dụng `dotenv` tự động trong `app.py`).
2. Chạy lệnh:
   ```
   python app.py
   ```
3. Mở trình duyệt và truy cập `http://localhost:5000`.

## Cấu trúc thư mục
```
KySu-1/
├── app.py                 # Ứng dụng Flask chính
├── CLAUDE.md              # Hướng dẫn viết mã
├── README.md              # Tài liệu này
├── requirements.txt       # Danh sách gói phụ thuộc
├── .env                   # Biến môi trường (không commit)
├── .gitignore             # Loại bỏ file nhạy cảm và dữ liệu lớn
├── data/                  # Dữ liệu đầu vào (CSV, PDF)
├── models/                # File mô hình AI và scaler
├── src/                   # Mã nguồn các module hỗ trợ
│   ├── admission_predictor.py
│   ├── ai_expert.py
│   ├── career_dataset.py
│   ├── dataset_generator.py
│   ├── holland_mapper.py
│   ├── subject_mapper.py
│   ├── student_vector.py
│   ├── transformer_model.py
│   └── university_recommender.py
├── templates/             # File HTML Flask
└── tmp/                   # Thư mục tạm (nếu có)
```

## Tác giả
- Nhóm phát triển KySu-1

## Giấy phép
Dự án này được phát hành dưới giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết (nếu có).
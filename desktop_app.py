import webview
import threading
import time
import sys
import os
import requests
from app import app  # Import ứng dụng Flask từ file app.py

def start_flask():
    """Hàm khởi chạy server Flask ngầm"""
    # Tắt chế độ reloader vì nó sẽ gây lỗi khi chạy đa luồng trong ứng dụng desktop
    app.run(port=5000, debug=False, use_reloader=False)

def wait_for_flask(timeout=30):
    """Chờ Flask server sẵn sàng với cơ chế thử lại và exponential backoff"""
    start_time = time.time()
    delay = 0.5  # Bắt đầu với 0.5 giây

    while time.time() - start_time < timeout:
        try:
            # Thử kết nối đến endpoint sức khỏe đơn giản
            response = requests.get("http://127.0.0.1:5000", timeout=1)
            # Nếu nhận được phản hồi (dù là lỗi 4xx/5xx), server đang chạy
            if response.status_code < 600:  # Mọi status code < 600 chỉ ra rằng server đang lắng nghe
                return True
        except requests.exceptions.RequestException:
            # Server chưa sẵn sàng, đợi và thử lại
            pass

        time.sleep(delay)
        delay = min(delay * 1.5, 2.0)  # Tăng delay dần lên đến max 2 giây

    return False

def run_desktop():
    """Hàm khởi chạy cửa sổ Desktop"""
    # 1. Chạy Flask trong một luồng riêng (Thread)
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # 2. Chờ Flask sẵn sàng với timeout và retry
    print("Đang khởi động máy chủ Flask...")
    if not wait_for_flask(timeout=30):
        print("Lỗi: Không thể kết nối đến Flask server sau 30 giây")
        print("Vui lòng kiểm tra:")
        print("   - Các dependencies đã được cài đặt đầy đủ?")
        print("   - Có đang chạyanother instance trên port 5000?")
        print("   - File app.py có tồn tại và không có lỗi cú pháp?")
        sys.exit(1)

    print("Flask server đã sẵn sàng! Đang khởi động giao diện desktop...")

    # 3. Tạo cửa sổ ứng dụng
    # Bạn có thể tùy chỉnh tiêu đề (title) và kích thước (width, height)
    window = webview.create_window(
        title='Hệ Thống Tư Vấn Hướng Nghiệp Đại Học - AI Driven',
        url='http://127.0.0.1:5000',
        width=1280,
        height=800,
        resizable=True,
        confirm_close=True
    )

    # 4. Bắt đầu ứng dụng (vòng lặp chính)
    webview.start()

    # 5. Thông báo khi application được đóng (Flask thread sẽ tự động dừng vì là daemon)
    print("Ứng dụng desktop đã được đóng.")

if __name__ == '__main__':
    print("--- Đang khởi động ứng dụng Desktop ---")
    print("Vui lòng chờ trong giây lát...")
    try:
        run_desktop()
    except KeyboardInterrupt:
        print("\nĐang đóng ứng dụng...")
        sys.exit(0)
    except Exception as e:
        print(f"\nLỗi không mong muốn xảy ra: {e}")
        sys.exit(1)

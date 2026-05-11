import webview
import threading
import time
import sys
import os
from app import app  # Import ứng dụng Flask từ file app.py

def start_flask():
    """Hàm khởi chạy server Flask ngầm"""
    # Tắt chế độ reloader vì nó sẽ gây lỗi khi chạy đa luồng trong ứng dụng desktop
    app.run(port=5000, debug=False, use_reloader=False)

def run_desktop():
    """Hàm khởi chạy cửa sổ Desktop"""
    # 1. Chạy Flask trong một luồng riêng (Thread)
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # 2. Chờ một chút để server kịp khởi động
    time.sleep(2)

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

if __name__ == '__main__':
    print("--- Đang khởi động ứng dụng Desktop ---")
    print("Vui lòng chờ trong giây lát...")
    try:
        run_desktop()
    except KeyboardInterrupt:
        print("\nĐang đóng ứng dụng...")
        sys.exit(0)

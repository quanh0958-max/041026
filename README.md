# 📈 Hệ Thống Kiểm Định Chiến Lược SMA + OBV & Tối Ưu Hóa Danh Mục (MPT vs Equal Weight)

Ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit** nhằm kiểm định tính hiệu quả định lượng của chiến lược giao dịch kết hợp giữa **Tín hiệu Xu hướng (SMA - Simple Moving Average)** và **Tín hiệu Dòng tiền/Khối lượng (OBV - On-Balance Volume)** trên thị trường chứng khoán Việt Nam (dữ liệu sàn HOSE 2020 - 2023).

---

## 🌟 Tính Năng Nổi Bật

1. **📊 Sàng lọc Định lượng Cổ phiếu (Train 2020 - 2021)**:
   - Chấm điểm và xếp hạng toàn bộ cổ phiếu trên HOSE dựa trên 4 tiêu chí cốt lõi:
     - 30% Tỷ suất sinh lời năm hóa (Annual Return)
     - 30% Tỷ số Sharpe (Sharpe Ratio)
     - 25% Thanh khoản bình quân ngày (Trading Value)
     - 15% Mức độ sụt giảm tối đa (Max Drawdown)
   - Tự động gợi ý và hiển thị 5 mã cổ phiếu tiêu biểu đa dạng ngành: **DIG, DGC, VND, HAH, MSN**.

2. **📈 Kiểm định & Trực quan hóa Tín hiệu Kỹ thuật**:
   - Tín hiệu giao dịch kết hợp linh hoạt theo cơ chế **OR** (hoặc **AND**).
   - Tránh thiên lệch nhìn trước tương lai (*Look-ahead bias*) với cơ chế khớp lệnh **T+1**.
   - Hỗ trợ khấu trừ phí giao dịch / trượt giá thực tế.
   - Biểu đồ tương tác đa tầng (**Plotly**): Đường giá, SMA ngắn/dài, điểm Mua/Bán, chỉ báo OBV & MA(OBV), cùng đường tăng trưởng vốn (*Equity Curve*).

3. **⚡ Tối ưu hóa Tham số (Hyperopt & Sliders)**:
   - Tinh chỉnh tham số thủ công với thanh trượt mượt mà hoặc bấm nút chạy thuật toán **Hyperopt (TPE)** để tìm bộ tham số tối ưu Sharpe trên tập dữ liệu Train.

4. **💼 Phân bổ Danh mục MPT (Markowitz) vs Equal Weight**:
   - Tối ưu hóa trọng số danh mục tối đa hóa Sharpe (*Long-only SLSQP*) trên tập Train.
   - Kiểm định ngoài mẫu (**Out-of-sample Test 2022**) so sánh 3 danh mục:
     1. *Equal Weight Buy & Hold* (Benchmark thụ động chia đều 1/N).
     2. *SMA + OBV Equal Weight*.
     3. *SMA + OBV MPT Optimal*.

5. **📑 Báo cáo & Kết luận Tự động**:
   - Tự động đánh giá hiệu quả phòng vệ vốn và quản trị rủi ro trong giai đoạn thị trường downtrend/khủng hoảng (năm 2022).
   - Hỗ trợ xuất bảng dữ liệu thống kê ra file CSV.

---

## 📁 Cấu Trúc Thư Mục

```text
├── app.py                   # Mã nguồn ứng dụng web Streamlit chính
├── requirements.txt         # Danh sách thư viện Python cần thiết
├── README.md                # Tài liệu hướng dẫn sử dụng và triển khai
└── HOSE_2020_2023_in.csv    # File dữ liệu lịch sử giá & khối lượng HOSE
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Chuẩn bị môi trường Python (khuyên dùng Python 3.9 - 3.11)
```bash
# Mở Terminal / PowerShell và di chuyển vào thư mục dự án
cd sma_obv_streamlit

# (Khuyến nghị) Tạo môi trường ảo
python -m venv venv
# Kích hoạt trên Windows:
venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Web App
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Triển Khai Lên Streamlit Community Cloud (Miễn Phí)

### Bước 1: Khởi tạo Git và Đẩy lên GitHub
1. Tạo một repository mới trên GitHub (ví dụ: `sma-obv-stock-backtest`).
2. Mở Terminal tại thư mục chứa 3 file (`app.py`, `requirements.txt`, `README.md`) và file dữ liệu `HOSE_2020_2023_in.csv`:
```bash
git init
git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
git commit -m "Khởi tạo Web App Kiểm định Chiến lược SMA + OBV và MPT"
git branch -M main
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/sma-obv-stock-backtest.git
git push -u origin main
```

### Bước 2: Deploy trên Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
2. Nhấn nút **"New app"**.
3. Điền thông tin:
   - **Repository**: Chọn repo bạn vừa tạo (`<tai-khoan>/sma-obv-stock-backtest`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Nhấn **"Deploy!"**. Streamlit Cloud sẽ tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy web app trong khoảng 1-2 phút.

---

## ⚙️ Yêu Cầu Định Dạng Dữ Liệu Tải Lên (Nếu dùng file riêng)

Nếu bạn muốn kiểm định với mã cổ phiếu hoặc sàn khác, file CSV cần có các cột tối thiểu:
- `date`: Định dạng ngày (VD: `YYYY-MM-DD` hoặc `DD/MM/YYYY`)
- `ticker`: Mã cổ phiếu (VD: `SSI`, `VND`, `FPT`...)
- `open`: Giá mở cửa
- `high`: Giá cao nhất
- `low`: Giá thấp nhất
- `close`: Giá đóng cửa
- `volume`: Khối lượng giao dịch

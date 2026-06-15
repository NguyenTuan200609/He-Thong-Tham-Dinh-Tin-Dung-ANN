import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import sqlite3
import os
from datetime import datetime

# Cấu hình trang ứng dụng
st.set_page_config(page_title="Hệ thống Thẩm định Tín dụng ANN", layout="wide")

# Xác định đường dẫn thư mục hiện tại để tránh lỗi đường dẫn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

# ==========================================
# 1. QUẢN LÝ ĐĂNG NHẬP VÀ PHÂN QUYỀN (GIAO DIỆN MỚI THÚ VỊ)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_role'] = None
    st.session_state['username'] = ""

def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_role'] = None
    st.session_state['username'] = ""
    st.rerun()

# Giao diện đăng nhập nâng cấp bằng CSS tùy biến
if not st.session_state['logged_in']:
    # Inject CSS tạo khối Card Đăng nhập độc đáo
    st.markdown("""
        <style>
        /* Bo nền ẩn bớt các thành phần thừa của Streamlit khi chưa đăng nhập */
        [data-testid="stSidebar"] {display: none;}
        
        .login-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border: 1px solid #334155;
            color: #f8fafc;
            margin-bottom: 20px;
        }
        .login-header {
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        .login-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Căn lề Form vào chính giữa trang bằng hệ thống cột [Trái, Giữa, Phải]
    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    
    with col_center:
        # Khối Card giao diện HTML
        st.markdown("""
            <div class="login-container">
                <h2 class="login-header">🛡️ RISK CONTROL CENTRAL</h2>
                <div class="login-subtitle">Hệ Thống Thẩm Định Tín Dụng Nội Bộ - Ngân Hàng Số</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Các Input đặt ngay bên dưới khối Card tạo cảm giác liền mạch
        username = st.text_input("👤 Mã định danh Nhân viên (Username)")
        password = st.text_input("🔑 Mật khẩu truy cập (Password)", type="password")
        role_select = st.selectbox("🎯 Khu vực làm việc chỉ định (Role)", ["Nhân viên tín dụng", "Quản lý rủi ro"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Nút bấm đăng nhập lớn, nổi bật
        if st.button("XÁC THỰC VÀ TRUY CẬP HỆ THỐNG 🔓", type="primary", use_container_width=True):
            if username == "officer" and password == "123":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = "Credit Officer"
                st.session_state['username'] = "Nguyễn Văn A (Nhân viên)"
                st.rerun()
            elif username == "manager" and password == "123":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = "Risk Manager"
                st.session_state['username'] = "Trần Thị B (Trưởng phòng)"
                st.rerun()
            else:
                st.sidebar.empty() # Reset lỗi ẩn
                st.error("❌ Mã nhân viên hoặc mật khẩu cấp phép không đúng!")
                
        # Khối gợi ý tài khoản được thiết kế gọn gàng, tinh tế
        with st.expander("🔑 Cấp quyền truy cập nhanh (Tài khoản thử nghiệm)"):
            st.code("• Nhân viên: officer / 123\n• Trưởng phòng: manager / 123", language="text")
            
    st.stop() # Chặn không cho xem các Tab bên dưới nếu chưa đăng nhập thành công

# Thanh bên điều hướng thông tin tài khoản (Chỉ xuất hiện SAU KHI ĐĂNG NHẬP)
with st.sidebar:
    st.markdown(f"### 📑 Xin chào, \n**{st.session_state['username']}**")
    st.markdown(f" Chức vụ: `{st.session_state['user_role']}`")
    st.markdown("---")
    if st.button("↩️ Đăng xuất", type="secondary", use_container_width=True):
        logout()

# ==========================================
# 2. KHỞI TẠO CƠ SỞ DỮ LIỆU (SQLITE)
# ==========================================
def init_db():
    db_path = os.path.join(CURRENT_DIR, 'credit_history.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS CreditPredictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thu_nhap REAL,
            tai_san INTEGER,
            nguoi_phu_thuoc INTEGER,
            tra_cham INTEGER,
            so_tien_vay REAL,
            thoi_han INTEGER,
            hoc_van TEXT,
            viec_lam TEXT,
            xac_suat REAL,
            ket_luan TEXT,
            ngay_du_bao TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_prediction(thu_nhap, tai_san, nguoi_phu_thuoc, tra_cham, so_tien_vay, thoi_han, hoc_van, viec_lam, prob, ket_luan):
    try:
        db_path = os.path.join(CURRENT_DIR, 'credit_history.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO CreditPredictions 
            (thu_nhap, tai_san, nguoi_phu_thuoc, tra_cham, so_tien_vay, thoi_han, hoc_van, viec_lam, xac_suat, ket_luan, ngay_du_bao)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (float(thu_nhap), int(tai_san), int(nguoi_phu_thuoc), int(tra_cham), float(so_tien_vay), int(thoi_han), hoc_van, viec_lam, float(prob), ket_luan, now))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Lỗi lưu trữ lịch sử: {e}")

# ==========================================
# 3. TẢI TÀI NGUYÊN MÔ HÌNH ML
# ==========================================
@st.cache_resource
def load_resources():
    model_path = os.path.join(CURRENT_DIR, 'best_credit_model.keras')
    scaler_path = os.path.join(CURRENT_DIR, 'scaler.pkl')
    columns_path = os.path.join(CURRENT_DIR, 'columns.pkl')
    
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(columns_path, 'rb') as f:
        model_columns = pickle.load(f)
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_resources()
except Exception as e:
    st.error("❌ Không tìm thấy mô hình! Hãy đảm bảo bạn đã chạy file 'train.py' trước.")
    st.stop()

# ==========================================
# 4. HÀM KIỂM TRA CHỐNG LỖI DỮ LIỆU (VALIDATOR)
# ==========================================
def validate_data(df):
    errors = []
    if (df['thu_nhap'] < 0).any():
        errors.append("Lỗi: Thu nhập hàng năm không được phép nhỏ hơn 0.")
    if (df['so_tien_vay'] < 0).any():
        errors.append("Lỗi: Số tiền đăng ký vay không được nhỏ hơn 0.")
    if (df['tai_san'] < 0).any() or (df['nguoi_phu_thuoc'] < 0).any() or (df['tra_cham'] < 0).any():
        errors.append("Lỗi: Các chỉ số số lượng (tài sản, người phụ thuộc, trả chậm) không được là số âm.")
        
    valid_edu = ["Cao đẳng", "Đại học", "Sau đại học"]
    valid_job = ["Có việc làm", "Thất nghiệp", "Tự do"]
    
    invalid_edu_rows = df[~df['hoc_van'].isin(valid_edu)]
    if not invalid_edu_rows.empty:
        errors.append(f"Lỗi: Cột 'hoc_van' chứa giá trị không hợp lệ. Chỉ chấp nhận: {valid_edu}")
        
    invalid_job_rows = df[~df['viec_lam'].isin(valid_job)]
    if not invalid_job_rows.empty:
        errors.append(f"Lỗi: Cột 'viec_lam' chứa giá trị không hợp lệ. Chỉ chấp nhận: {valid_job}")
        
    return errors

# ==========================================
# 5. ĐỊNH NGHĨA GIAI ĐOẠN GIAO DIỆN (TABS)
# ==========================================
st.title("📊 Hệ thống Phân tích & Dự đoán Nguy cơ Vỡ nợ Tín dụng")

if st.session_state['user_role'] == "Risk Manager":
    tabs = st.tabs(["👤 Dự Đoán Đơn Lẻ", "📂 Dự Đoán Hàng Loạt", "📜 Lịch Sử & Dashboard Hệ Thống"])
else:
    tabs = st.tabs(["👤 Dự Đoán Đơn Lẻ", "📂 Dự Đoán Hàng Loạt"])
    st.warning("⚠️ Tài khoản nhân viên của bạn bị hạn chế quyền truy cập Tab dữ liệu lịch sử nội bộ ngân hàng.")

# ------------------------------------------
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ------------------------------------------
with tabs[0]:
    st.subheader("Nhập thông tin khách hàng đơn lẻ")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 💰 Thông tin tài chính")
        thu_nhap = st.number_input("Thu nhập hàng năm ($)", min_value=-50000, max_value=500000, value=45000, step=1000)
        so_tien_vay = st.number_input("Số tiền muốn vay ($)", min_value=500, max_value=200000, value=15000, step=500)
        thoi_han = st.number_input("Thời hạn vay (tháng)", min_value=3, max_value=120, value=36, step=1)
        tai_san = st.number_input("Số lượng tài sản sở hữu", min_value=0, max_value=20, value=1, step=1)

    with col2:
        st.markdown("##### 👤 Thông tin cá nhân & Lịch sử")
        nguoi_phu_thuoc = st.number_input("Số người phụ thuộc", min_value=0, max_value=10, value=0, step=1)
        tra_cham = st.number_input("Số lần trả chậm trong lịch sử", min_value=0, max_value=30, value=0, step=1)
        hoc_van = st.selectbox("Trình độ học vấn", ["Cao đẳng", "Đại học", "Sau đại học"])
        viec_lam = st.selectbox("Tình trạng việc làm", ["Có việc làm", "Thất nghiệp", "Tự do"])

    st.markdown("---")

    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH RỦI RO", type="primary", use_container_width=True):
        single_df = pd.DataFrame([{
            'thu_nhap': thu_nhap, 'tai_san': tai_san, 'nguoi_phu_thuoc': nguoi_phu_thuoc,
            'tra_cham': tra_cham, 'so_tien_vay': so_tien_vay, 'thoi_han': thoi_han,
            'hoc_van': hoc_van, 'viec_lam': viec_lam
        }])
        
        data_errors = validate_data(single_df)
        if data_errors:
            for err in data_errors: st.error(err)
            st.warning("❌ Hủy bỏ quy trình chạy mạng ANN do dữ liệu đầu vào không hợp lệ!")
        else:
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)
            input_data['thu_nhap'] = thu_nhap
            input_data['tai_san'] = tai_san
            input_data['nguoi_phu_thuoc'] = nguoi_phu_thuoc
            input_data['tra_cham'] = tra_cham
            input_data['so_tien_vay'] = so_tien_vay
            input_data['thoi_han'] = thoi_han
            
            col_hoc_van = f"hoc_van_{hoc_van}"
            col_viec_lam = f"viec_lam_{viec_lam}"
            if col_hoc_van in input_data.columns: input_data[col_hoc_van] = 1
            if col_viec_lam in input_data.columns: input_data[col_viec_lam] = 1
                
            num_cols = ['thu_nhap', 'tai_san', 'nguoi_phu_thuoc', 'tra_cham', 'so_tien_vay', 'thoi_han']
            input_data[num_cols] = scaler.transform(input_data[num_cols])
            
            prob = model.predict(input_data)[0][0]
            
            st.subheader("📊 Kết quả phân tích từ mạng ANN:")
            st.progress(float(prob))
            st.write(f"**Xác suất xảy ra vỡ nợ:** `{prob * 100:.2f}%`")
            
            ket_luan = "Từ chối cho vay (Nguy cơ cao)" if prob > 0.5 else "Phê duyệt khoản vay (An toàn)"
            if prob > 0.5:
                st.error(f"🚨 **CẢNH BÁO:** {ket_luan}")
            else:
                st.success(f"✅ **XÁC NHẬN:** {ket_luan}")
                
            save_prediction(thu_nhap, tai_san, nguoi_phu_thuoc, tra_cham, so_tien_vay, thoi_han, hoc_van, viec_lam, prob, ket_luan)

# ------------------------------------------
# TAB 2: DỰ ĐOÁN HÀNG LOẠT
# ------------------------------------------
with tabs[1]:
    st.subheader("Xử lý hồ sơ tín dụng hàng loạt từ File")
    uploaded_file = st.file_uploader("Kéo thả file dữ liệu khách hàng (.csv, .xlsx)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Dữ liệu xem trước:")
            st.dataframe(df_input.head(3))
            
            required_cols = ['thu_nhap', 'tai_san', 'nguoi_phu_thuoc', 'tra_cham', 'so_tien_vay', 'thoi_han', 'hoc_van', 'viec_lam']
            missing_cols = [c for c in required_cols if c not in df_input.columns]
            
            if missing_cols:
                st.error(f"❌ File tải lên thiếu các cột bắt buộc: {missing_cols}")
            else:
                if st.button("⚡ CHẠY XỬ LÝ DỰ ĐOÁN HÀNG LOẠT"):
                    file_errors = validate_data(df_input)
                    if file_errors:
                        st.markdown("### ❌ Lỗi cấu trúc dữ liệu trong File:")
                        for err in file_errors: st.error(err)
                        st.error("🚨 Vui lòng sửa lại file Excel đúng quy định trước khi chạy AI.")
                    else:
                        X_batch = pd.DataFrame(0, index=range(len(df_input)), columns=model_columns)
                        num_cols = ['thu_nhap', 'tai_san', 'nguoi_phu_thuoc', 'tra_cham', 'so_tien_vay', 'thoi_han']
                        for col in num_cols: X_batch[col] = df_input[col]
                        
                        for i in range(len(df_input)):
                            col_hv = f"hoc_van_{df_input.loc[i, 'hoc_van']}"
                            col_vl = f"viec_lam_{df_input.loc[i, 'viec_lam']}"
                            if col_hv in X_batch.columns: X_batch.loc[i, col_hv] = 1
                            if col_vl in X_batch.columns: X_batch.loc[i, col_vl] = 1
                        
                        X_batch[num_cols] = scaler.transform(X_batch[num_cols])
                        batch_probs = model.predict(X_batch).flatten()
                        
                        df_output = df_input.copy()
                        df_output['Xác suất vỡ nợ (%)'] = np.round(batch_probs * 100, 2)
                        df_output['Quyết định khuyến nghị'] = np.where(batch_probs > 0.5, "Từ chối cho vay (Nguy cơ cao)", "Phê duyệt khoản vay (An toàn)")
                        
                        st.subheader("📋 Bảng tổng hợp kết quả:")
                        st.dataframe(df_output)
                        
                        for idx, row in df_output.iterrows():
                            save_prediction(row['thu_nhap'], row['tai_san'], row['nguoi_phu_thuoc'], row['tra_cham'], row['so_tien_vay'], row['thoi_han'], row['hoc_van'], row['viec_lam'], batch_probs[idx], row['Quyết định khuyến nghị'])
                        
                        csv_data = df_output.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(label="📥 TẢI VỀ KẾT QUẢ (.CSV)", data=csv_data, file_name="ket_qua_batch.csv", mime="text/csv", use_container_width=True)
        except Exception as err:
            st.error(f"Lỗi đọc file: {err}")

# ------------------------------------------
# TAB 3: LỊCH SỬ & DASHBOARD
# ------------------------------------------
if st.session_state['user_role'] == "Risk Manager":
    with tabs[2]:
        st.subheader("Biểu đồ phân tích rủi ro & Nhật ký hệ thống")
        db_path = os.path.join(CURRENT_DIR, 'credit_history.db')
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            df_history = pd.read_sql_query("SELECT * FROM CreditPredictions ORDER BY id DESC", conn)
            conn.close()
            
            if not df_history.empty:
                import plotly.express as px
                
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    fig_pie = px.pie(df_history, names="ket_luan", title="Tỷ lệ Quyết định phê duyệt tín dụng", hole=0.4,
                                     color_discrete_map={"Phê duyệt khoản vay (An toàn)": "#2ECC71", "Từ chối cho vay (Nguy cơ cao)": "#E74C3C"})
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with chart_col2:
                    fig_bar = px.histogram(df_history, x="ket_luan", y="thu_nhap", histfunc="avg", title="Mức thu nhập trung bình theo quyết định",
                                           color="ket_luan", color_discrete_map={"Phê duyệt khoản vay (An toàn)": "#2ECC71", "Từ chối cho vay (Nguy cơ cao)": "#E74C3C"})
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")
                if st.button("🗑️ Xóa sạch cơ sở dữ liệu lịch sử", type="secondary"):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM CreditPredictions")
                    conn.commit()
                    conn.close()
                    st.rerun()
                
                df_display = df_history.copy()
                df_display.columns = ["ID", "Thu nhập", "Tài sản", "Người phụ thuộc", "Trả chậm", "Số tiền vay", "Thời hạn", "Học vấn", "Việc làm", "Xác suất vỡ nợ", "Kết luận", "Thời gian"]
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("Cơ sở dữ liệu đang trống.")
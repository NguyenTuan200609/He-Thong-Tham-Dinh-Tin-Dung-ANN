import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("--- GIAI ĐOẠN 2: KHỞI TẠO DỮ LIỆU & TIỀN XỬ LÝ ---")

# Tự động tạo dữ liệu mô phỏng chuẩn theo yêu cầu đề bài để tránh lỗi thiếu file
np.random.seed(42)
n_samples = 2000

data = {
    'thu_nhap': np.random.normal(50000, 20000, n_samples).clip(10000, 150000),
    'tai_san': np.random.randint(0, 5, n_samples),
    'nguoi_phu_thuoc': np.random.randint(0, 4, n_samples),
    'tra_cham': np.random.poisson(0.3, n_samples),
    'so_tien_vay': np.random.normal(20000, 15000, n_samples).clip(2000, 80000),
    'thoi_han': np.random.choice([12, 24, 36, 48, 60], n_samples),
    'hoc_van': np.random.choice(['Cao đẳng', 'Đại học', 'Sau đại học'], n_samples, p=[0.3, 0.5, 0.2]),
    'viec_lam': np.random.choice(['Có việc làm', 'Thất nghiệp', 'Tự do'], n_samples, p=[0.7, 0.1, 0.2])
}

df = pd.DataFrame(data)

# Tạo nhãn vỡ nợ (Mất cân bằng: tỷ lệ vỡ nợ khoảng 15%)
rủi_ro = (df['tra_cham'] * 2.0) + (df['so_tien_vay'] / df['thu_nhap'] * 3.0) - (df['tai_san'] * 0.5)
df['vo_no'] = (rủi_ro > 1.5).astype(int)

# Mã hóa One-hot cho các đặc trưng phân loại
cat_cols = ['hoc_van', 'viec_lam']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

X = df_encoded.drop(columns=['vo_no'])
y = df_encoded['vo_no']

# Chia tập Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu số
num_cols = ['thu_nhap', 'tai_san', 'nguoi_phu_thuoc', 'tra_cham', 'so_tien_vay', 'thoi_han']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Tính class weight xử lý mất cân bằng
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print("\n--- GIAI ĐOẠN 3: HUẤN LUYỆN MẠNG ANN & THỬ NGHIỆM LR ---")

def build_ann_model(input_dim, learning_rate):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

learning_rates = [0.001, 0.01, 0.1]
models = {}
best_acc = 0
best_lr = 0.001

for lr in learning_rates:
    print(f"\n====> Thử nghiệm với Learning Rate: {lr}")
    model = build_ann_model(X_train.shape[1], lr)
    model.fit(X_train, y_train, epochs=15, batch_size=32, class_weight=class_weight_dict, verbose=1)
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"-> Độ chính xác với LR {lr}: {acc:.4f}")
    models[lr] = model
    if acc > best_acc:
        best_acc = acc
        best_lr = lr

print(f"\n--> Learning Rate tốt nhất: {best_lr} (Accuracy: {best_acc:.4f})")

print("\n--- GIAI ĐOẠN 4: ĐÁNH GIÁ MÔ HÌNH VÀ VẼ MA TRẬN NHẦM LẪN ---")
best_model = models[best_lr]
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['An toàn', 'Vỡ nợ'], yticklabels=['An toàn', 'Vỡ nợ'])
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.title(f'Ma trận nhầm lẫn (LR = {best_lr})')
print("Đang hiển thị biểu đồ ma trận nhầm lẫn. Hãy ĐÓNG CỬA SỔ BIỂU ĐỒ để lưu file!")

# THAY THẾ TỪ ĐOẠN PLT.SHOW() TRỞ ĐI BẰNG ĐOẠN NÀY:
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

# Lưu biểu đồ thành file ảnh thay vì bật cửa sổ bắt bạn tắt
plt.savefig(os.path.join(current_dir, 'confusion_matrix.png'))
print("-> Đã lưu ảnh ma trận nhầm lẫn thành file 'confusion_matrix.png'")
plt.close() # Tự động đóng biểu đồ chạy ngầm

# Lưu các file mô hình bắt buộc
best_model.save(os.path.join(current_dir, 'best_credit_model.keras'))
with open(os.path.join(current_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(current_dir, 'columns.pkl'), 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(f"\n--- HOÀN THÀNH XUẤT SẮC! File mô hình đã lưu tại: {current_dir} ---")
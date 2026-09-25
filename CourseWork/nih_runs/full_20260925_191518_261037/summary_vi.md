### Kết quả của nhóm


Sử dụng 5604 ảnh sau kiểm tra/chọn mẫu;
train/validation/test lần lượt 3956/824/824.
Chọn **E3** bằng validation MAE, không bằng test.
Test MAE **11.43 năm**, RMSE **14.21 năm**, MSE **201.96 năm²**,
R² **0.316**, bias **0.58 năm**.
MAE giảm so với train-mean baseline **2.68 năm** (âm nghĩa là kém hơn baseline).
Khoảng bootstrap theo bệnh nhân cho MAE: **[10.75, 12.16] năm**.

Đây là kết quả trên Random Sample NIH, không phải toàn bộ NIH ChestX-ray14.
Một seed và một split chưa đủ để khẳng định cấu hình tối ưu; mô hình không dùng trong lâm sàng.

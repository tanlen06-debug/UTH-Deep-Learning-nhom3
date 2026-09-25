# 🩻 Nhóm 3 — Medical Image Regression

## Dự đoán tuổi bệnh nhân từ ảnh X-quang ngực

Coursework học phần **Deep Learning**, thực hiện bài toán **hồi quy tuổi** bằng mô hình **CNN + Global Average Pooling + Linear output**.

Nhóm sử dụng **Random Sample of NIH Chest X-ray Dataset**, so sánh **MAE và MSE loss**, đồng thời kiểm tra ảnh hưởng của **data augmentation** thông qua bốn thí nghiệm E1–E4.

---

## 1. Thông tin dự án

| Nội dung | Thông tin |
|---|---|
| Nhóm thực hiện | Nhóm 3 |
| Học phần | Deep Learning |
| Bài toán | Medical Image Regression |
| Mục tiêu | Dự đoán tuổi bệnh nhân từ ảnh X-quang ngực |
| Đầu vào | Một ảnh X-quang ngực |
| Đầu ra | Một giá trị tuổi liên tục, đơn vị năm |
| Mô hình | CNN + Global Average Pooling + Linear |
| Framework | PyTorch |
| Môi trường | Jupyter Notebook trong Visual Studio Code |
| Notebook chính | [NIH_Age_Regression_Nhom3_Local.ipynb](./NIH_Age_Regression_Nhom3_Local.ipynb) |

> **Phạm vi:** Nhóm thực hiện hồi quy tuổi trên tập mẫu NIH. Kết quả không đại diện cho toàn bộ NIH ChestX-ray14 và chưa được xác nhận cho sử dụng lâm sàng.

---

## 2. Mục tiêu và câu hỏi nghiên cứu

### Mục tiêu

- Xây dựng pipeline Deep Learning hoàn chỉnh cho bài toán hồi quy ảnh y tế.
- Sử dụng nhãn `Patient Age` để dự đoán tuổi.
- Chia dữ liệu theo bệnh nhân nhằm hạn chế rò rỉ dữ liệu.
- Triển khai đúng kiến trúc CNN + GAP + Linear.
- So sánh hai hàm mất mát MAE và MSE.
- Đánh giá tác động của augmentation.
- Phân tích kết quả tổng thể và sai số theo nhóm tuổi.

### Câu hỏi nghiên cứu

1. CNN có tốt hơn baseline dự đoán tuổi trung bình hoặc trung vị không?
2. MAE loss và MSE loss ảnh hưởng thế nào đến kết quả?
3. Augmentation có cải thiện khả năng tổng quát hóa không?
4. Mô hình hoạt động tốt hoặc kém ở những nhóm tuổi nào?

---

## 3. Dataset

**Dữ liệu sử dụng:**

[Random Sample of NIH Chest X-ray Dataset — Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample)

**Nguồn dữ liệu gốc trong đề bài:**

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Metadata cần thiết

| Trường | Vai trò |
|---|---|
| `Image Index` | Ghép metadata với tên ảnh |
| `Patient ID` | Chia dữ liệu theo bệnh nhân |
| `Patient Age` | Nhãn hồi quy tuổi |

Nhóm chọn **dự đoán tuổi** vì dữ liệu có nhãn `Patient Age`. Các nhãn bệnh không được dùng thay cho tuổi hoặc kích thước khối u.

### Kết quả kiểm tra dữ liệu trong lần chạy final

| Hạng mục | Số lượng |
|---|---:|
| Dòng metadata ban đầu | 5.606 |
| Dòng bị loại khi làm sạch | 2 |
| Ảnh sử dụng | 5.604 |
| Bệnh nhân | 4.228 |
| Ảnh thiếu hoặc không đọc được ghi nhận | 0 |

Tuổi được chuyển về đơn vị năm và kiểm tra trong phạm vi nghiên cứu từ **1 đến 100 năm**.

> Ví dụ: `018M` tương ứng **1,5 năm**, không phải 18 tuổi.

---

## 4. Tổ chức tệp

| Đường dẫn | Nội dung |
|---|---|
| `CourseWork/README.md` | Giới thiệu và hướng dẫn thực hiện |
| `CourseWork/00_COURSEWORK_PLAN.md` | Kế hoạch coursework |
| `CourseWork/01_MEMBER_TASKS.md` | Phân công thành viên |
| `CourseWork/NIH_Age_Regression_Nhom3_Local.ipynb` | Notebook chính |
| `CourseWork/archive (5)/` | Ví dụ thư mục dữ liệu đã giải nén |
| `CourseWork/nih_runs/` | Các lần chạy và kết quả đã lưu |

Dữ liệu ảnh và metadata được tải riêng từ Kaggle. Có thể đặt dữ liệu ngoài repository rồi cập nhật `DATA_DIR`.

Thư mục dữ liệu cần chứa:

- File `sample_labels.csv`.
- Các ảnh PNG trong thư mục con.

Notebook tìm metadata và ảnh bên dưới `DATA_DIR`, nên thư mục giải nén có thể có thêm một cấp `sample/`.

---

## 5. Hướng dẫn chạy trên VS Code

### Bước 1 — Chuẩn bị môi trường

Cài Python cùng hai extension của Microsoft trong VS Code:

- **Python**
- **Jupyter**

Mở thư mục repository và chọn đúng môi trường Python làm kernel của notebook.

Các thư viện sử dụng:

```text
torch
torchvision
numpy
pandas
matplotlib
pillow
scikit-learn
tqdm
ipykernel
```

Cài PyTorch phù hợp với CPU hoặc GPU theo hướng dẫn:

https://pytorch.org/get-started/locally/

### Bước 2 — Tải và giải nén dataset

Tải bộ dữ liệu từ Kaggle, sau đó giải nén.

> `DATA_DIR` phải trỏ đến **thư mục đã giải nén**, không trỏ đến file ZIP.

### Bước 3 — Chỉnh cell cấu hình

Ví dụ:

```python
from pathlib import Path

DATA_DIR = Path(r"D:\Datasets\NIH_sample")
CSV_FILE = None

MODE = "check"
RUN_DIR = None

OUT_ROOT = Path.cwd() / "nih_runs"
```

Thay đường dẫn ví dụ bằng đường dẫn thực tế trên máy.

Nếu có nhiều file CSV cùng tên nhưng khác nội dung, chỉ định rõ:

```python
CSV_FILE = Path(r"D:\Datasets\NIH_sample\sample_labels.csv")
```

### Bước 4 — Chọn chế độ chạy

| Chế độ | Mục đích |
|---|---|
| `check` | Kiểm tra dữ liệu, split, preprocessing và mô hình; không train E1–E4 |
| `smoke` | Chạy nhanh với tập nhỏ và 1 epoch để kiểm tra pipeline |
| `full` | Huấn luyện đầy đủ E1–E4 |
| `reload` | Đọc lại kết quả đã lưu mà không train lại |

**Quy trình khuyến nghị:**

1. Chạy `check` để xác nhận dữ liệu và đường dẫn.
2. Chạy `smoke` nếu cần kiểm tra toàn bộ pipeline.
3. Chạy `full` để tạo kết quả chính thức.
4. Dùng `reload` để xem lại kết quả khi chuẩn bị báo cáo.

> Kết quả `smoke` chỉ dùng kiểm tra code, không dùng làm kết quả final.

Sau khi thay đổi cấu hình, chọn **Restart Kernel → Run All**.

### Bước 5 — Huấn luyện đầy đủ

```python
MODE = "full"

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

LR = 1e-3
WEIGHT_DECAY = 1e-4
AGE_SCALE = 100.0
NUM_WORKERS = 0
```

Notebook tự sử dụng CUDA nếu khả dụng, nếu không sẽ chạy CPU.

### Bước 6 — Xem lại kết quả đã lưu

```python
MODE = "reload"

RUN_DIR = Path(
    r"D:\Projects\UTH-Deep-Learning-nhom3"
    r"\CourseWork\nih_runs\full_20260925_191518_261037"
)
```

Thay `RUN_DIR` bằng thư mục thực tế của lần chạy cần xem.

Cần giữ dataset tương ứng cùng các tệp cấu hình, split, history, predictions, bảng kết quả và checkpoint. Giữ cấu hình tiền xử lý phù hợp với lần chạy đã lưu.

> `reload` là đọc lại kết quả và kiểm tra inference. Đây không phải chức năng tiếp tục huấn luyện từ epoch bị ngắt.

---

## 6. Quy trình xử lý

| Bước | Nội dung |
|---|---|
| 1 | Đọc metadata và ghép đường dẫn ảnh |
| 2 | Chuyển tuổi về năm, kiểm tra dữ liệu |
| 3 | Khám phá phân bố tuổi và số ảnh mỗi bệnh nhân |
| 4 | Chia train/validation/test theo bệnh nhân |
| 5 | Tiền xử lý ảnh và chuẩn hóa nhãn |
| 6 | Xây dựng CNN + GAP + Linear |
| 7 | Huấn luyện bốn thí nghiệm E1–E4 |
| 8 | Chọn checkpoint theo validation MAE |
| 9 | Đánh giá test và so sánh baseline |
| 10 | Phân tích sai số, lưu kết quả và báo cáo |

### Phân chia dữ liệu trong lần chạy final

| Tập dữ liệu | Số ảnh | Số bệnh nhân |
|---|---:|---:|
| Train | 3.956 | 2.959 |
| Validation | 824 | 634 |
| Test | 824 | 635 |

Notebook kiểm tra:

- Không có bệnh nhân xuất hiện ở nhiều tập.
- Không có hash nội dung ảnh trùng nhau giữa các tập.

### Tiền xử lý

- Chuyển ảnh sang grayscale.
- Resize giữ tỷ lệ và padding về **224 × 224**.
- Chuyển ảnh thành tensor và normalize.
- Chia tuổi cho 100 khi tối ưu.
- Nhân lại 100 trước khi tính metric theo đơn vị năm.

---

## 7. Kiến trúc mô hình

Mô hình gồm bốn convolution block:

```text
Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

| Thành phần | Kích thước đầu ra cho một ảnh |
|---|---|
| Input | 1 × 224 × 224 |
| Conv block 1 | 32 × 112 × 112 |
| Conv block 2 | 64 × 56 × 56 |
| Conv block 3 | 128 × 28 × 28 |
| Conv block 4 | 256 × 14 × 14 |
| Global Average Pooling | 256 × 1 × 1 |
| Flatten | 256 |
| Linear | 1 |

**Số tham số huấn luyện: 389.057.**

Các lớp cuối:

```python
self.gap = nn.AdaptiveAvgPool2d(1)
self.output = nn.Linear(256, 1)
```

Đầu ra là một giá trị liên tục. Mô hình không sử dụng softmax hoặc sigmoid ở lớp cuối.

---

## 8. Thiết kế thí nghiệm E1–E4

| Thí nghiệm | Loss | Augmentation |
|---|---|---|
| E1 | MSE | Không |
| E2 | MAE / L1 | Không |
| E3 | MSE | Có |
| E4 | MAE / L1 | Có |

Các thí nghiệm giữ cùng:

- Patient-level split.
- Kiến trúc mô hình.
- Seed 42.
- Batch size 16.
- AdamW, learning rate 0,001.
- Weight decay 0,0001.
- Ngân sách 10 epoch.

### Augmentation

```python
transforms.RandomAffine(
    degrees=7,
    translate=(0.03, 0.03),
    scale=(0.95, 1.05),
    fill=0
)
```

Augmentation chỉ áp dụng ngẫu nhiên trên **train**. Validation và test sử dụng preprocessing xác định.

**Nguyên tắc lựa chọn mô hình:**

- Trong mỗi thí nghiệm: lấy checkpoint có validation MAE thấp nhất.
- Giữa E1–E4: chọn cấu hình có validation MAE thấp nhất.
- Test dùng để đánh giá, không dùng để chọn epoch hoặc điều chỉnh cấu hình.

---

## 9. Kết quả final

### Kết quả validation

| Thí nghiệm | Epoch tốt nhất | Validation MAE — năm |
|---|---:|---:|
| E1 | 9 | 11,323900 |
| E2 | 9 | 11,790808 |
| **E3** | **10** | **11,258113** |
| E4 | 9 | 12,569042 |

**Cấu hình được chọn: E3 — MSE có augmentation.**

### Kết quả test

| Mô hình | MAE ↓ | RMSE ↓ | R² ↑ | Bias |
|---|---:|---:|---:|---:|
| E1 | 11,695 | 14,469 | 0,291 | −5,678 |
| E2 | 11,828 | 14,685 | 0,270 | −2,809 |
| **E3** | **11,432** | **14,211** | **0,316** | **+0,576** |
| E4 | 12,592 | 15,395 | 0,198 | −4,559 |
| Baseline tuổi trung bình train | 14,107 | 17,187 | ≈0 | +0,028 |
| Baseline tuổi trung vị train | 13,942 | 17,366 | −0,021 | +2,488 |

MAE, RMSE và bias có đơn vị **năm**. R² không có đơn vị.

### Nhận xét

- E3 giảm MAE khoảng **2,675 năm**, tương đương **19%**, so với baseline tuổi trung bình.
- Augmentation cải thiện kết quả khi dùng MSE trong lần chạy này.
- Augmentation làm tăng sai số khi dùng MAE trong lần chạy này.
- E3 chỉ tốt hơn E1 khoảng **0,066 năm trên validation**; cần chạy nhiều seed để kiểm tra độ ổn định.
- Mô hình có xu hướng dự đoán cao ở người trẻ và thấp ở người lớn tuổi.

> Các kết luận trên dựa trên một seed, một split và 10 epoch cho mỗi thí nghiệm.

---

## 10. Đánh giá và trực quan hóa

Notebook cung cấp:

- Phân bố tuổi.
- Phân bố tuổi giữa train, validation và test.
- Ảnh trước và sau augmentation.
- Training/validation loss.
- Training/validation MAE.
- Bảng so sánh E1–E4 với baseline.
- Biểu đồ predicted vs. actual.
- Phân bố residual.
- Sai số theo nhóm tuổi.
- Ví dụ có sai số nhỏ và sai số lớn.
- Bootstrap theo bệnh nhân.
- Kiểm tra nạp checkpoint và dự đoán một ảnh.

### Một số kết quả phân tích E3

| Chỉ số | Kết quả |
|---|---|
| MSE | 201,957 năm² |
| Median absolute error | 9,789 năm |
| P90 absolute error | 22,961 năm |
| Khoảng tin cậy bootstrap 95% của MAE | 10,75–12,16 năm |
| Số lần bootstrap | 1.000 |
| MAE nhóm trên 40 đến 60 tuổi | 6,093 năm |
| MAE nhóm trên 80 đến 100 tuổi | 29,020 năm; chỉ có 6 ảnh |

Khoảng bootstrap là khoảng cho **MAE tổng hợp**, không phải khoảng dự đoán tuổi của từng bệnh nhân.

Vì đây là bài toán Regression, nhóm dùng sai số hồi quy để đánh giá. Accuracy, confusion matrix và false positive/false negative không thay thế các metric hồi quy trên.

---

## 11. Lưu kết quả

Mỗi lần chạy tạo thư mục riêng trong `nih_runs`.

| Tệp | Nội dung |
|---|---|
| `config.json` | Cấu hình và thông tin môi trường |
| `train_split.csv` | Manifest tập train |
| `val_split.csv` | Manifest tập validation |
| `test_split.csv` | Manifest tập test |
| `data_audit.csv` | Kết quả kiểm tra dữ liệu |
| `E1_history.csv` … `E4_history.csv` | Lịch sử huấn luyện |
| `E1_best.pth` … `E4_best.pth` | Checkpoint tốt nhất |
| `validation_results.csv` | Kết quả validation |
| `test_results.csv` | Kết quả test |
| `E1_predictions.csv` … `E4_predictions.csv` | Dự đoán và sai số |
| `figures/` | Biểu đồ |
| `summary_vi.md` | Tóm tắt kết quả |

Không ghép checkpoint, split và metric từ các lần chạy khác nhau.

---

## 12. Làm việc nhóm trên GitHub

### Quy trình

1. Xác định nhiệm vụ trong Issue hoặc `01_MEMBER_TASKS.md`.
2. Tạo branch riêng.
3. Chỉnh sửa code hoặc tài liệu.
4. Kiểm tra phần thay đổi và output liên quan.
5. Commit những tệp cần thiết.
6. Mở Pull Request để thành viên khác review.
7. Merge sau khi thống nhất.

### Ví dụ cập nhật README

Chạy từ thư mục gốc repository sau khi đã lưu các thay đổi đang làm:

```bash
git switch main
git pull --ff-only
git switch -c docs/coursework-readme

git diff -- CourseWork/README.md
git add CourseWork/README.md
git commit -m "docs: update regression coursework README"

git push -u origin docs/coursework-readme
```

Sau đó mở Pull Request từ branch mới vào `main`.

### Nguyên tắc quản lý

- Thống nhất người tích hợp notebook final để giảm xung đột.
- Giữ output thực tế trong bản nộp báo cáo.
- Ghi rõ cấu hình khi thay đổi thí nghiệm.
- Tải dataset riêng, không đưa toàn bộ ảnh và ZIP vào repository.
- Ghi nhận đóng góp thực tế trong `01_MEMBER_TASKS.md`.
- Kiểm tra mọi số liệu do AI hỗ trợ diễn giải bằng output thật.

---

## 13. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Không thấy `sample_labels.csv` | Giải nén dữ liệu và kiểm tra `DATA_DIR` |
| CSV thiếu `Patient Age` | Kiểm tra đúng metadata có nhãn tuổi |
| Đường dẫn lặp `CourseWork/CourseWork` | Kiểm tra `Path.cwd()` hoặc dùng đường dẫn tuyệt đối |
| Thiếu thư viện | Cài vào đúng môi trường Python của kernel |
| Không chạy GPU | Kiểm tra PyTorch và `torch.cuda.is_available()` |
| Train quá lâu khi chuẩn bị báo cáo | Dùng output đã lưu hoặc chế độ `reload` |
| Reload báo dữ liệu không khớp | Kiểm tra metadata, ảnh và manifest của đúng run |
| Không tìm thấy checkpoint | Kiểm tra `RUN_DIR` và file `<experiment>_best.pth` |

---

## 14. Kết luận

Nhóm đã triển khai pipeline **hồi quy tuổi từ ảnh X-quang ngực** theo yêu cầu CNN + Global Average Pooling + Linear output, đồng thời so sánh MAE/MSE và kiểm tra augmentation bằng E1–E4.

Trong lần chạy final, **E3 — MSE có augmentation** đạt test MAE **11,432 năm**, RMSE **14,211 năm** và R² **0,316**, cải thiện MAE khoảng **19%** so với baseline tuổi trung bình.

Mô hình vẫn còn sai số lớn ở hai đầu phân bố tuổi. Hướng phát triển tiếp theo là chạy nhiều seed, điều chỉnh trên validation và đánh giá trên dữ liệu độc lập.

---

## 15. Tài liệu tham khảo

- [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [PyTorch — hướng dẫn cài đặt](https://pytorch.org/get-started/locally/)
- [Jupyter Notebooks trong VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- Notebook final: `NIH_Age_Regression_Nhom3_Local.ipynb`.
- Run được báo cáo: `full_20260925_191518_261037`.

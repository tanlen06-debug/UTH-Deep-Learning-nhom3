Nhóm 3 — Medical Image Regression

Dự đoán tuổi bệnh nhân từ ảnh X-quang ngực

Coursework học phần Deep Learning, triển khai bài toán hồi quy tuổi bằng CNN + Global Average Pooling + Linear output trên Random Sample of NIH Chest X-ray Dataset. Nhóm so sánh MAE với MSE loss và đánh giá ảnh hưởng của data augmentation bằng bốn thí nghiệm E1–E4.

Thông tin

Nội dung

Repository

UTH-Deep-Learning-nhom3

Notebook chính

NIH_Age_Regression_Nhom3_Local.ipynb

Loại bài toán

Regression — dự đoán một giá trị tuổi liên tục

Đầu vào

Một ảnh X-quang ngực, chuyển thành ảnh xám

Đầu ra

Tuổi dự đoán, đơn vị năm

Framework

PyTorch

Môi trường thực hiện

Jupyter Notebook trong Visual Studio Code

Kết quả chính của lần chạy final

E3: test MAE 11,432 năm, RMSE 14,211 năm, R² 0,316

Kết quả trong README được trích từ output của notebook final ở chế độ full, seed 42, 10 epoch cho mỗi thí nghiệm. Đây là kết quả trên tập mẫu NIH, không đại diện cho toàn bộ NIH ChestX-ray14. Tên notebook dùng trong repository được thống nhất là NIH_Age_Regression_Nhom3_Local.ipynb.

1. Mục tiêu và yêu cầu đề bài

Đề bài yêu cầu Medical Image Regression (Age or Tumor Size Prediction): dự đoán một đại lượng liên tục từ ảnh y tế, sử dụng CNN, Global Average Pooling và đầu ra tuyến tính; mở rộng bằng so sánh MAE/MSE và thử augmentation.

Nhóm chọn dự đoán tuổi vì metadata của tập mẫu có trường Patient Age. Nhãn tuổi là mục tiêu học; Patient ID chỉ phục vụ chia dữ liệu và phân tích theo bệnh nhân. Các nhãn bệnh không được dùng thay cho tuổi hoặc kích thước khối u.

Các câu hỏi thực nghiệm:

CNN có dự đoán tuổi tốt hơn baseline luôn trả về tuổi trung bình hoặc trung vị của tập train không?

MAE loss và MSE loss tạo ra kết quả khác nhau như thế nào trên cùng điều kiện thực nghiệm?

Augmentation có cải thiện khả năng tổng quát hóa không, và tác động có giống nhau giữa hai loại loss không?

Sai số thay đổi ra sao theo tuổi, và những trường hợp nào mô hình dự đoán kém?

Yêu cầu

Cách triển khai

Dữ liệu NIH ChestX-ray14

Dùng tập mẫu có ảnh và metadata tuổi để giảm chi phí lưu trữ, huấn luyện

Giá trị đầu ra liên tục

Một số thực biểu diễn tuổi

CNN

Bốn khối Conv2d → BatchNorm2d → ReLU → MaxPool2d

Global Average Pooling

nn.AdaptiveAvgPool2d(1)

Linear output

nn.Linear(256, 1)

So sánh loss

nn.MSELoss() và nn.L1Loss()

Thử augmentation

So sánh có/không RandomAffine trên train

Đánh giá hồi quy

MAE, MSE, RMSE, R², bias và phân tích sai số

2. Dữ liệu và cách tải

Tập mẫu sử dụng: Random Sample of NIH Chest X-ray Dataset — Kaggle.

Nguồn dữ liệu gốc trong đề: NIH ChestX-ray14.

Metadata cần thiết: sample_labels.csv.

Các trường bắt buộc: Image Index, Patient ID, Patient Age.

Tải bộ dữ liệu từ Kaggle rồi giải nén. DATA_DIR phải trỏ đến thư mục đã giải nén, không trỏ đến file .zip. Có thể đặt dữ liệu ngoài repository; chỉ cần cập nhật đường dẫn trong cell cấu hình.

Notebook tìm sample_labels.csv và ảnh trong các thư mục con. Nếu có nhiều CSV cùng tên nhưng khác nội dung, cần chỉ định CSV_FILE để tránh đọc nhầm.

Số liệu kiểm tra trong lần chạy final

Hạng mục

Kết quả

Dòng metadata ban đầu

5.606

Dòng bị loại trong bước làm sạch

2

Ảnh sử dụng sau kiểm tra

5.604

Bệnh nhân

4.228

Ảnh thiếu hoặc không đọc được ghi nhận

0

Ảnh trùng nội dung bị loại

0

Tuổi trung bình / trung vị

46,71 / 49 năm

Nhãn tuổi được đổi về năm; ví dụ 018M tương ứng 1,5 năm. Notebook kiểm tra phạm vi nghiên cứu từ 1 đến 100 năm. Việc loại mẫu được ghi nhận để có thể kiểm tra lại.

3. Tổ chức tệp trong CourseWork

Các đường dẫn dưới đây mô tả cách bố trí để chạy notebook. Thư mục dữ liệu được tải riêng; thư mục kết quả được tạo khi chạy, không phải tất cả đều bắt buộc có sẵn trên GitHub.

Đường dẫn

Vai trò

CourseWork/README.md

Tổng quan, hướng dẫn chạy và kết quả

CourseWork/NIH_Age_Regression_Nhom3_Local.ipynb

Notebook chính để thực hiện và báo cáo

CourseWork/00_COURSEWORK_PLAN.md

Kế hoạch coursework

CourseWork/01_MEMBER_TASKS.md

Phân công và theo dõi đóng góp

CourseWork/archive (5)/sample_labels.csv

Một vị trí có thể đặt metadata sau giải nén

CourseWork/archive (5)/sample/images/

Một vị trí có thể đặt ảnh PNG

CourseWork/nih_runs/full_<timestamp>/

Kết quả một lần chạy đầy đủ

Tên archive (5) chỉ là tên thư mục tải xuống trên máy thực hiện. Có thể đổi tên hoặc đặt ở ổ đĩa khác rồi cập nhật DATA_DIR. Metadata nằm sâu hơn một cấp vẫn được tìm thấy nếu nằm bên dưới DATA_DIR.

4. Cài đặt và mở notebook trong VS Code

4.1. Lấy mã nguồn

Nếu máy chưa có repository, chạy trong terminal:

git clone https://github.com/tanlen06-debug/UTH-Deep-Learning-nhom3.git
cd UTH-Deep-Learning-nhom3
code .

Nếu đã có mã nguồn, mở thư mục hiện tại bằng VS Code. Kiểm tra và lưu các thay đổi đang làm trước khi cập nhật từ GitHub.

4.2. Tạo môi trường Python trên Windows

Cài extension Python và Jupyter của Microsoft trong VS Code. Ví dụ dưới đây dùng Python 3.11; cần cài phiên bản này trước khi dùng py -3.11.

Chạy tại thư mục gốc repository:

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib pillow scikit-learn tqdm ipykernel

Cài PyTorch theo phần cứng:

CPU trên Windows: có thể dùng lệnh bên dưới.

GPU NVIDIA: lấy lệnh cài phù hợp từ trang PyTorch chính thức, rồi chạy bằng Python của .venv. Không chọn phiên bản CUDA chỉ theo tên GPU.

.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Các lệnh này là cách thiết lập môi trường mới, không phải bản khóa phiên bản của lần chạy final. Khi cần tái lập, đối chiếu config.json của lần chạy đã lưu và ghi lại các phiên bản thư viện sử dụng.

4.3. Chọn đúng kernel

Mở notebook, chọn Select Kernel → Python Environments → .venv. Có thể kiểm tra bằng:

import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

Notebook hiện chọn CUDA khi khả dụng, nếu không sẽ chạy CPU. Có thể xem các output đã lưu trong notebook mà không cần huấn luyện lại.

5. Cấu hình và các chế độ chạy

Sửa các biến trong cell cấu hình đầu tiên có DATA_DIR và MODE. Ví dụ đường dẫn tuyệt đối trên Windows:

from pathlib import Path

DATA_DIR = Path(r"D:\Datasets\NIH_sample")
CSV_FILE = None
MODE = "check"
RUN_DIR = None
OUT_ROOT = Path.cwd() / "nih_runs"

Thay đường dẫn ví dụ bằng thư mục thực tế của máy. Path.cwd() là thư mục làm việc của kernel; kiểm tra dòng Working directory để biết nih_runs sẽ được tạo ở đâu.

MODE

Mục đích

Cách sử dụng

check

Kiểm tra dữ liệu, split, transforms và mô hình; không train E1–E4

Chạy đầu tiên khi thiết lập trên máy mới

smoke

Kiểm tra nhanh pipeline với tập nhỏ và 1 epoch mỗi thí nghiệm

Chỉ kiểm tra code; không dùng kết quả làm báo cáo final

full

Huấn luyện E1–E4 theo toàn bộ cấu hình

Dùng để tạo kết quả thực nghiệm chính thức

reload

Đọc split, lịch sử và kết quả đã lưu; dựng lại phần đánh giá

Dùng để xem lại một lần chạy, tránh train lại

Sau khi đổi chế độ hoặc cấu hình, chọn Restart Kernel → Run All để tránh sử dụng nhầm biến từ lần chạy trước. Lưu bản notebook final có output trước khi chạy thử cấu hình mới.

Huấn luyện đầy đủ

MODE = "full"
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
AGE_SCALE = 100.0
NUM_WORKERS = 0

Notebook dùng NUM_WORKERS = 0 để thuận tiện khi chạy Jupyter trên Windows. Nếu thay batch size, image size, số epoch hoặc split để giảm thời gian chạy, phải ghi đó là cấu hình khác; không gộp kết quả mới vào bảng final cũ.

Mở lại kết quả mà không train lại

MODE = "reload"
RUN_DIR = Path(r"D:\Projects\UTH-Deep-Learning-nhom3\CourseWork\nih_runs\full_20260925_191518_261037")

Thay RUN_DIR bằng thư mục thực tế. Cần giữ metadata và ảnh tương ứng, config.json, các split CSV, history CSV, prediction CSV, bảng validation/test và checkpoint. Cell cuối kiểm tra inference còn cần file E3_best.pth hoặc checkpoint của cấu hình được chọn.

Notebook đối chiếu hash metadata và manifest trước khi khôi phục split. Khi reload, giữ các thiết lập tiền xử lý tương ứng với config.json đã lưu. Chế độ này không phải tiếp tục huấn luyện từ epoch bị ngắt: notebook chưa lưu trạng thái optimizer để resume chính xác.

6. Quy trình Deep Learning trong notebook

Đọc và kiểm tra dữ liệu: ghép filename với metadata, đổi tuổi về năm, kiểm tra ảnh và nội dung trùng.

EDA: phân bố tuổi, số ảnh mỗi bệnh nhân và kích thước ảnh.

Chia theo bệnh nhân: cùng bệnh nhân chỉ thuộc một trong train, validation hoặc test.

Tiền xử lý: grayscale, resize giữ tỷ lệ và padding về 224 × 224, chuyển tensor, normalize.

Chuẩn hóa nhãn: tuổi chia 100 trong lúc tối ưu; nhân lại 100 khi tính metric theo năm.

Xây dựng CNN + GAP + Linear: đầu ra một giá trị liên tục.

Huấn luyện E1–E4: cùng split, seed và ngân sách epoch.

Chọn checkpoint: validation MAE nhỏ nhất trong từng thí nghiệm; chọn cấu hình tốt nhất cũng bằng validation MAE.

Đánh giá test: so sánh với baseline, phân tích residual, nhóm tuổi và ví dụ sai số.

Lưu và trình bày: giữ cấu hình, bảng kết quả, hình, checkpoint và diễn giải giới hạn của thí nghiệm.

Phân chia dữ liệu trong lần chạy final

Tập

Số ảnh

Số bệnh nhân

Tuổi trung bình

Train

3.956

2.959

46,54

Validation

824

634

47,73

Test

824

635

46,51

Tỷ lệ chia mục tiêu khoảng 70% / 15% / 15% theo bệnh nhân. Số ảnh không nhất thiết đúng tỷ lệ này vì mỗi người có số ảnh khác nhau. Notebook xác nhận không giao Patient ID và không giao hash nội dung ảnh giữa ba tập.

7. Mô hình và thiết kế thí nghiệm

Kiến trúc

Thành phần

Kích thước đầu ra cho một ảnh

Ảnh đầu vào

1 × 224 × 224

Conv block 1

32 × 112 × 112

Conv block 2

64 × 56 × 56

Conv block 3

128 × 28 × 28

Conv block 4

256 × 14 × 14

Global Average Pooling

256 × 1 × 1

Flatten và Linear

1 giá trị

Mô hình có 389.057 tham số huấn luyện. GAP giảm chiều không gian trước lớp Linear. Đầu ra không bị giới hạn bởi sigmoid hoặc softmax.

Ma trận E1–E4

Thí nghiệm

Loss

Augmentation trên train

E1

MSE

Không

E2

MAE / L1

Không

E3

MSE

Có

E4

MAE / L1

Có

Augmentation dùng RandomAffine: xoay tối đa ±7°, dịch chuyển tối đa 3% theo mỗi trục, scale trong khoảng 0,95–1,05. Validation và test dùng preprocessing xác định, không dùng augmentation ngẫu nhiên.

Các phép so sánh chính: E1/E2 và E3/E4 cho tác động của loss; E1/E3 và E2/E4 cho tác động của augmentation. MSE và MAE loss có thang đo khác nhau; dùng validation MAE theo năm làm tiêu chí chung để chọn mô hình.

8. Kết quả thực nghiệm final

Lựa chọn mô hình bằng validation

Thí nghiệm

Epoch tốt nhất

Validation MAE (năm)

E1

9

11,323900

E2

9

11,790808

E3

10

11,258113

E4

9

12,569042

E3 được chọn trước khi diễn giải kết quả test. Test không dùng để chọn epoch hoặc điều chỉnh hyperparameter.

Kết quả trên 824 ảnh test

Mô hình

MAE ↓ (năm)

RMSE ↓ (năm)

R² ↑

Bias (năm)

E1

11,695

14,469

0,291

−5,678

E2

11,828

14,685

0,270

−2,809

E3

11,432

14,211

0,316

+0,576

E4

12,592

15,395

0,198

−4,559

Baseline tuổi trung bình train

14,107

17,187

≈0

+0,028

Baseline tuổi trung vị train

13,942

17,366

−0,021

+2,488

Trong lần chạy này, E3 giảm MAE khoảng 2,675 năm, tương đương 19%, so với baseline tuổi trung bình. Augmentation cải thiện test MAE khi dùng MSE nhưng làm tăng sai số khi dùng MAE. Chênh lệch validation giữa E3 và E1 chỉ khoảng 0,066 năm, nên cần nhiều lần chạy để đánh giá độ ổn định của thứ hạng.

Cách đọc metric và hình ảnh

MAE: trung bình độ lớn sai số; MAE 11,432 nghĩa là lệch trung bình khoảng 11,432 năm trên tập test này.

MSE / RMSE: nhạy hơn với các sai số lớn; MSE có đơn vị năm², RMSE có đơn vị năm.

R²: so sánh tổng sai số bình phương với biến thiên nhãn; không phải accuracy phần trăm.

Bias: trung bình prediction − actual; dương là dự đoán tuổi cao hơn, âm là thấp hơn.

Learning curves: quan sát tiến trình train/validation; không so sánh trực tiếp trị số MSE loss với MAE loss.

Predicted vs. actual: điểm càng gần đường chéo càng gần nhãn thật.

Residual plots: kiểm tra mô hình có dự đoán lệch theo tuổi hay không.

Ví dụ sai số nhỏ/lớn: minh họa trường hợp cụ thể; đây là các mẫu được chọn theo sai số, không phải mẫu ngẫu nhiên đại diện.

Phân tích sâu E3

Phân tích

Kết quả

MSE

201,957 năm²

Median absolute error

9,789 năm

P90 absolute error

22,961 năm

Khoảng tin cậy bootstrap 95% của MAE

10,75–12,16 năm

Số lần bootstrap

1.000; lấy mẫu theo bệnh nhân

Nhóm tuổi có MAE thấp nhất

Trên 40 đến 60 tuổi: 6,093 năm

Nhóm trên 80 đến 100 tuổi

MAE 29,020 năm; chỉ 6 ảnh

Mô hình có xu hướng dự đoán cao ở người trẻ và thấp ở người lớn tuổi. Bias tổng gần 0 có thể do hai chiều sai số triệt tiêu nhau. Khoảng bootstrap là khoảng cho MAE tổng hợp, không phải khoảng dự đoán tuổi của từng người và không phản ánh biến động do train lại bằng seed khác.

Vì đầu ra là tuổi liên tục, phần đánh giá sử dụng sai số hồi quy. Không dùng confusion matrix, false positive/false negative hoặc accuracy của bài toán phân loại để thay cho các metric trên.

9. Tệp kết quả và tái lập

Mỗi lần chạy tạo thư mục riêng bên dưới OUT_ROOT. Các tệp quan trọng gồm:

Tệp

Nội dung

config.json

Seed, hyperparameter, môi trường và hash metadata

train_split.csv, val_split.csv, test_split.csv

Manifest ảnh/bệnh nhân của từng tập

data_audit.csv, excluded_metadata.csv

Kết quả kiểm tra và metadata bị loại

image_issues.csv, duplicate_images.csv

Vấn đề ảnh và trùng lặp được ghi nhận

E1_history.csv … E4_history.csv

Loss và metric theo epoch

E1_best.pth … E4_best.pth

Trọng số checkpoint tốt nhất theo validation

validation_results.csv, test_results.csv

Bảng kết quả tổng hợp

E1_predictions.csv … E4_predictions.csv

Nhãn thật, dự đoán và sai số trên test

figures/

Các biểu đồ được lưu bởi cell đánh giá

summary_vi.md

Tóm tắt định lượng bằng tiếng Việt

Giữ nguyên các tệp của cùng một run. Không ghép bảng metric của run này với checkpoint hoặc split của run khác. Seed cố định hỗ trợ tái lập nhưng không bảo đảm kết quả trùng tuyệt đối giữa mọi phần cứng và phiên bản thư viện.

10. Quy trình làm việc nhóm trên GitHub

Cách đóng góp

Xác định nhiệm vụ trong 01_MEMBER_TASKS.md hoặc tạo Issue mô tả rõ việc cần làm.

Từ bản main đã cập nhật, tạo branch riêng cho từng thay đổi.

Sửa nội dung, kiểm tra cell liên quan và ghi rõ cấu hình đã dùng.

Commit những tệp cần thiết, kiểm tra diff trước khi push.

Mở Pull Request, nêu vấn đề, thay đổi và cách kiểm tra; nhờ thành viên khác review trước khi merge.

Ví dụ chỉ cập nhật README, chạy từ thư mục gốc repository sau khi working tree đã sạch:

git switch main
git pull --ff-only
git switch -c docs/coursework-readme
git status
git diff -- CourseWork/README.md
git add CourseWork/README.md
git commit -m "docs: update regression coursework README"
git push -u origin docs/coursework-readme

Sau đó tạo Pull Request từ docs/coursework-readme vào main. Nếu sửa trực tiếp trên GitHub, mở CourseWork/README.md, chọn biểu tượng bút chì, thay nội dung, xem Preview rồi lưu thay đổi vào branch phù hợp.

Phân công tham khảo

Phần việc

Đầu ra cần kiểm tra

Điều phối và tích hợp

Mục tiêu, tiến độ và bản final thống nhất

Dữ liệu

Metadata, audit và patient-level split

Mô hình

CNN + GAP + Linear, input/output shape

Loss và training

E1–E4, learning curves và checkpoint

Augmentation

Phép biến đổi train và so sánh có đối chứng

Đánh giá và báo cáo

Metric, error analysis, README, report và slide

Tên thành viên và đóng góp thực tế cần ghi trong 01_MEMBER_TASKS.md. Bảng trên mô tả nhóm công việc, không thay cho minh chứng đóng góp.

Quản lý notebook và dữ liệu

Tránh để nhiều người cùng sửa một notebook trong cùng thời điểm; thống nhất người tích hợp bản final.

Giữ output thực tế trong notebook nộp báo cáo; bỏ output lỗi hoặc log tiến trình quá dài khi chúng không cần thiết.

Đưa mã nguồn, tài liệu, cấu hình và bảng kết quả cần đối chiếu lên GitHub.

Dữ liệu ảnh thô, ZIP và môi trường .venv được tải/cài riêng.

Nếu checkpoint được chia sẻ qua GitHub Releases hoặc nơi lưu trữ khác, ghi rõ run tương ứng để người khác có thể reload.

Có thể bổ sung các dòng sau vào .gitignore ở thư mục gốc nếu phù hợp với cách bố trí dữ liệu của nhóm:

.venv/
__pycache__/
.ipynb_checkpoints/
/CourseWork/archive*/
/CourseWork/**/*.zip
/CourseWork/nih_runs/**/*.pth

Các quy tắc này chỉ bỏ qua tệp chưa được Git theo dõi. Không tự động xóa tệp đã commit. Không bỏ qua toàn bộ nih_runs nếu nhóm muốn giữ CSV, cấu hình và biểu đồ để đối chiếu kết quả.

Sử dụng AI trong coursework

AI có thể hỗ trợ giải thích code, rà lỗi, viết tài liệu và gợi ý cách trực quan hóa. Nhóm chịu trách nhiệm kiểm tra code, chạy thực nghiệm và đối chiếu mọi con số với output thật. Khi AI hỗ trợ thay đổi phương pháp, cần ghi lại thay đổi và kiểm chứng trước khi đưa vào báo cáo. Không tạo số liệu thay cho kết quả chưa chạy.

11. Xử lý lỗi thường gặp

Hiện tượng

Cách kiểm tra và xử lý

Không thấy sample_labels.csv

Kiểm tra đã giải nén; đặt DATA_DIR đến thư mục chứa dữ liệu hoặc chỉ định CSV_FILE

Đường dẫn lặp CourseWork/CourseWork

In Path.cwd(); dùng đường dẫn tuyệt đối cho DATA_DIR

CSV thiếu Patient Age

Kiểm tra đúng metadata của tập mẫu; CSV chỉ có nhãn bệnh chưa cung cấp nhãn hồi quy tuổi

ModuleNotFoundError

Cài thư viện bằng Python của .venv, chọn lại kernel rồi restart

Không dùng được GPU

Kiểm tra phần cứng, bản PyTorch và torch.cuda.is_available(); notebook vẫn có thể chạy CPU

E1–E4 train quá lâu khi chuẩn bị báo cáo

Xem output đã lưu hoặc dùng reload với run đầy đủ; không đổi smoke thành kết quả final

reload báo dữ liệu không khớp

Kiểm tra metadata, ảnh và manifest của cùng run; không bỏ qua kiểm tra hash

Không tìm thấy checkpoint khi inference

Kiểm tra file <experiment>_best.pth trong đúng RUN_DIR

12. Kết luận và giới hạn

Nhóm đã xây dựng pipeline hồi quy tuổi từ ảnh X-quang: kiểm tra dữ liệu, chia theo bệnh nhân, tiền xử lý, CNN + GAP + Linear, so sánh MAE/MSE, thử augmentation và phân tích sai số.

Trong lần chạy final, E3 — MSE có augmentation đạt kết quả tốt nhất theo validation và có test MAE 11,432 năm, cải thiện khoảng 19% so với baseline tuổi trung bình. Tuy nhiên, kết quả còn phụ thuộc một seed, một split và ngân sách 10 epoch. Sai số lớn ở hai đầu phân bố tuổi và số lượng ít ở nhóm tuổi cao là những hạn chế cần trình bày rõ.

Hướng phát triển: lặp nhiều seed, kiểm tra độ ổn định của thứ hạng, điều chỉnh trên validation, cải thiện dữ liệu ở nhóm tuổi ít mẫu và đánh giá trên tập độc lập. Mô hình hiện phục vụ coursework, chưa được xác nhận cho sử dụng lâm sàng.

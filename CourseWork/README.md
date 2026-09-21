## 📂 Dataset

Do kích thước tập dữ liệu lớn, repository này không lưu trữ trực tiếp các file ảnh và dữ liệu thô. 

- **Tên tập dữ liệu:** [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- **Mô tả:** Tập mẫu ngẫu nhiên gồm 5.606 ảnh X-quang lồng ngực kèm nhãn và siêu dữ liệu (metadata) được trích xuất từ bộ dữ liệu gốc của Viện Y tế Quốc gia Hoa Kỳ (NIH).
- **Cách thiết lập dữ liệu để chạy Notebook:**
  1. Tải bộ dữ liệu từ liên kết Kaggle phía trên (nhấn nút **Download**).
  2. Giải nén toàn bộ tệp vào đường dẫn dự án theo cấu trúc:
     ```text
     CourseWork/
     └── archive (5)/
         ├── sample/
         │   └── images/ (chứa các file ảnh .png)
         └── sample_labels.csv
     ```
  3. Mở file `NIH_Age_Regression_Nhom3_Local.ipynb` và thực thi các cell tuần tự.
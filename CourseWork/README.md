# 🩻 Group 3 — Medical Image Regression

## Patient Age Prediction from Chest X-ray Images

This Deep Learning coursework develops a **patient age regression model** using **CNN + Global Average Pooling + Linear output**.

The project uses the **Random Sample of NIH Chest X-ray Dataset**, compares **MAE and MSE loss**, and evaluates **data augmentation** through four controlled experiments: E1–E4.

---

## 1. Project Overview

| Item | Description |
|---|---|
| Team | Group 3 |
| Course | Deep Learning |
| Task | Medical Image Regression |
| Objective | Predict patient age from a chest X-ray image |
| Input | One chest X-ray image |
| Output | A continuous age estimate in years |
| Architecture | CNN + Global Average Pooling + Linear |
| Framework | PyTorch |
| Environment | Jupyter Notebook in Visual Studio Code |
| Main notebook | [NIH_Age_Regression_Nhom3_Local.ipynb](./NIH_Age_Regression_Nhom3_Local.ipynb) |

> **Scope:** This project evaluates age regression on a sample of the NIH dataset. Its results do not represent performance on the complete NIH ChestX-ray14 dataset. The model has not been validated for clinical use.

---

## 2. Objectives and Research Questions

### Objectives

- Build a complete Deep Learning pipeline for medical image regression.
- Use `Patient Age` as the continuous target.
- Split the dataset by patient to reduce data leakage.
- Implement CNN + Global Average Pooling + Linear output.
- Compare MAE and MSE loss under consistent experimental conditions.
- Evaluate the effect of data augmentation.
- Analyze overall performance, age-dependent errors, and individual predictions.

### Research Questions

1. Can the CNN outperform constant predictions based on the training-set mean or median age?
2. How do MAE and MSE loss affect regression performance?
3. Does augmentation improve generalization?
4. Which age groups and individual cases produce the largest errors?

### Alignment with the Coursework Requirements

| Requirement | Implementation |
|---|---|
| Medical image regression | Predict patient age from chest X-ray images |
| NIH ChestX-ray14 data | Use the NIH random sample with age metadata |
| CNN | Four convolutional blocks |
| Global Average Pooling | `nn.AdaptiveAvgPool2d(1)` |
| Linear output | `nn.Linear(256, 1)` |
| Compare MAE and MSE | E1–E4 experimental matrix |
| Test augmentations | Compare training with and without random affine transformations |

---

## 3. Dataset

**Dataset used:**

[Random Sample of NIH Chest X-ray Dataset — Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/sample)

**Original dataset source specified in the coursework:**

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Required Metadata

| Field | Purpose |
|---|---|
| `Image Index` | Match metadata records to image filenames |
| `Patient ID` | Group examinations belonging to the same patient |
| `Patient Age` | Provide the regression target |

The project selects **age prediction** because the sample includes `Patient Age`. Disease labels are not used as substitutes for age or tumor size.

### Data Audit from the Final Run

| Item | Count |
|---|---:|
| Initial metadata records | 5,606 |
| Records excluded during cleaning | 2 |
| Images retained | 5,604 |
| Unique patients | 4,228 |
| Missing or unreadable images identified | 0 |

Age values are converted into years and checked against the study range of **1–100 years**.

> For example, `018M` represents **1.5 years**, not 18 years.

---

## 4. Project Files

| Path | Purpose |
|---|---|
| `CourseWork/README.md` | Project overview and instructions |
| `CourseWork/00_COURSEWORK_PLAN.md` | Coursework plan |
| `CourseWork/01_MEMBER_TASKS.md` | Team responsibilities |
| `CourseWork/NIH_Age_Regression_Nhom3_Local.ipynb` | Main notebook |
| `CourseWork/archive (5)/` | Example location for the extracted dataset |
| `CourseWork/nih_runs/` | Saved experiment runs and outputs |

Download the images and metadata separately from Kaggle. The dataset may also be stored outside the repository by updating `DATA_DIR`.

The extracted dataset must contain:

- `sample_labels.csv`.
- The corresponding PNG images in its subdirectories.

The notebook searches recursively under `DATA_DIR`. An additional nested `sample/` directory is therefore acceptable.

---

## 5. Running the Notebook in VS Code

### Step 1 — Prepare the Environment

Install Python and the following Microsoft extensions in VS Code:

- **Python**
- **Jupyter**

Open the repository folder and select the appropriate Python environment as the notebook kernel.

Required packages:

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

Install PyTorch according to the operating system and available hardware:

https://pytorch.org/get-started/locally/

### Step 2 — Download and Extract the Dataset

Download the dataset from Kaggle and extract the archive.

> `DATA_DIR` must point to an **extracted directory**, not a ZIP file.

### Step 3 — Update the Configuration Cell

Example:

```python
from pathlib import Path

DATA_DIR = Path(r"D:\Datasets\NIH_sample")
CSV_FILE = None

MODE = "check"
RUN_DIR = None

OUT_ROOT = Path.cwd() / "nih_runs"
```

Replace the example path with the actual dataset location.

If multiple CSV files have the same name but different contents, specify the intended file explicitly:

```python
CSV_FILE = Path(
    r"D:\Datasets\NIH_sample\sample_labels.csv"
)
```

Check the printed working directory to confirm where the output folder will be created.

### Step 4 — Select an Execution Mode

| Mode | Purpose |
|---|---|
| `check` | Validate data, splits, preprocessing, and model shapes without training E1–E4 |
| `smoke` | Test the pipeline using small subsets and one epoch per experiment |
| `full` | Train all four experiments using the configured training budget |
| `reload` | Load saved results and reproduce evaluation without retraining |

**Recommended workflow:**

1. Run `check` when setting up the project.
2. Run `smoke` if a short pipeline test is needed.
3. Run `full` to generate the main experimental results.
4. Use `reload` to review an existing run before presenting.

> Smoke-test results are intended for code verification and must not be presented as final experimental results.

After changing the configuration, select **Restart Kernel → Run All**.

### Step 5 — Run the Full Experiments

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

The notebook uses CUDA when available and otherwise runs on CPU.

Changing the image size, batch size, epoch count, or split creates a different experimental configuration. Record such changes before comparing results.

### Step 6 — Reload an Existing Run

```python
MODE = "reload"

RUN_DIR = Path(
    r"D:\Projects\UTH-Deep-Learning-nhom3"
    r"\CourseWork\nih_runs\full_20260925_191518_261037"
)
```

Replace `RUN_DIR` with the actual saved-run directory.

Reloading requires the matching dataset and saved configuration, split manifests, histories, predictions, result tables, and checkpoints. Keep preprocessing settings consistent with the saved run.

> `reload` restores results and supports inference checks. It does not resume interrupted training because the notebook does not save the optimizer state required for an exact training continuation.

---

## 6. Deep Learning Workflow

| Stage | Description |
|---|---|
| 1 | Load metadata and match image paths |
| 2 | Convert age labels into years and validate the data |
| 3 | Explore age distribution and repeated examinations |
| 4 | Create patient-level train, validation, and test splits |
| 5 | Preprocess images and scale the regression target |
| 6 | Build CNN + GAP + Linear |
| 7 | Train E1–E4 |
| 8 | Select checkpoints using validation MAE |
| 9 | Evaluate test performance and compare baselines |
| 10 | Analyze errors and save reproducible outputs |

### Final Data Split

| Subset | Images | Patients |
|---|---:|---:|
| Training | 3,956 | 2,959 |
| Validation | 824 | 634 |
| Test | 824 | 635 |

The notebook checks that:

- Patient IDs do not overlap between subsets.
- Identical image-content hashes do not overlap between subsets.

These checks reduce important leakage risks but do not detect every possible near-duplicate image.

### Preprocessing

- Convert images to grayscale.
- Resize while preserving aspect ratio and pad to **224 × 224**.
- Convert images into tensors and normalize.
- Divide age labels by 100 during optimization.
- Convert predictions back to years before calculating evaluation metrics.

---

## 7. Model Architecture

Each convolutional block contains:

```text
Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

| Component | Output shape for one image |
|---|---|
| Input | 1 × 224 × 224 |
| Convolutional block 1 | 32 × 112 × 112 |
| Convolutional block 2 | 64 × 56 × 56 |
| Convolutional block 3 | 128 × 28 × 28 |
| Convolutional block 4 | 256 × 14 × 14 |
| Global Average Pooling | 256 × 1 × 1 |
| Flatten | 256 |
| Linear output | 1 |

**Trainable parameters: 389,057.**

The final layers are:

```python
self.gap = nn.AdaptiveAvgPool2d(1)
self.output = nn.Linear(256, 1)
```

Global Average Pooling summarizes each feature map before the linear layer.

The model returns a continuous value. No softmax or sigmoid is applied to the final output.

---

## 8. Experimental Design

| Experiment | Loss | Training augmentation |
|---|---|---|
| E1 | MSE | No |
| E2 | MAE / L1 | No |
| E3 | MSE | Yes |
| E4 | MAE / L1 | Yes |

The experiments share:

- The same patient-level split.
- The same model architecture.
- Random seed 42.
- Batch size 16.
- AdamW optimizer.
- Learning rate 0.001.
- Weight decay 0.0001.
- Ten epochs per experiment.

### Augmentation

```python
transforms.RandomAffine(
    degrees=7,
    translate=(0.03, 0.03),
    scale=(0.95, 1.05),
    fill=0
)
```

Random augmentation is applied only to the **training set**. Validation and test images use deterministic preprocessing.

### Controlled Comparisons

- **Loss effect:** E1 vs. E2 and E3 vs. E4.
- **Augmentation effect:** E1 vs. E3 and E2 vs. E4.

MSE and MAE loss have different scales. Their raw loss magnitudes should not be compared directly. **Validation MAE in years** provides the common model-selection criterion.

### Model Selection

- Within each experiment, retain the checkpoint with the lowest validation MAE.
- Across E1–E4, select the configuration with the lowest validation MAE.
- Use test results for final evaluation, not for choosing epochs or tuning hyperparameters.

---

## 9. Final Results

### Validation Results

| Experiment | Best epoch | Validation MAE — years |
|---|---:|---:|
| E1 | 9 | 11.323900 |
| E2 | 9 | 11.790808 |
| **E3** | **10** | **11.258113** |
| E4 | 9 | 12.569042 |

**Selected configuration: E3 — MSE with augmentation.**

### Test Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Bias |
|---|---:|---:|---:|---:|
| E1 | 11.695 | 14.469 | 0.291 | −5.678 |
| E2 | 11.828 | 14.685 | 0.270 | −2.809 |
| **E3** | **11.432** | **14.211** | **0.316** | **+0.576** |
| E4 | 12.592 | 15.395 | 0.198 | −4.559 |
| Training-mean baseline | 14.107 | 17.187 | ≈0 | +0.028 |
| Training-median baseline | 13.942 | 17.366 | −0.021 | +2.488 |

MAE, RMSE, and bias are measured in **years**. R² is dimensionless.

Both baseline predictions are calculated exclusively from training-set ages.

### Interpretation

- E3 reduces MAE by approximately **2.675 years**, or **19%**, relative to the training-mean baseline.
- Augmentation improves test performance with MSE in this run.
- Augmentation increases test error with MAE in this run.
- E3 improves validation MAE over E1 by only approximately **0.066 years**.
- The model tends to overestimate younger ages and underestimate older ages.

> These findings come from one seed, one split, and ten epochs per experiment. Repeated runs are needed to assess the stability of the ranking.

---

## 10. Evaluation and Visualizations

The notebook includes:

- Overall age distribution.
- Age distributions across training, validation, and test subsets.
- Images before and after augmentation.
- Training and validation loss curves.
- Training and validation MAE curves.
- E1–E4 and baseline comparisons.
- Predicted-versus-actual age plots.
- Residual distributions.
- Error analysis by age group.
- Examples with small and large prediction errors.
- Patient-level bootstrap analysis.
- Checkpoint loading and single-image inference verification.

### Additional E3 Results

| Measure | Result |
|---|---|
| MSE | 201.957 years² |
| Median absolute error | 9.789 years |
| 90th-percentile absolute error | 22.961 years |
| 95% bootstrap interval for MAE | 10.75–12.16 years |
| Bootstrap repetitions | 1,000 |
| MAE for ages above 40 through 60 | 6.093 years |
| MAE for ages above 80 through 100 | 29.020 years; only 6 images |

Bootstrap resampling is performed by patient, keeping examinations from the same patient together.

The interval describes uncertainty in the **aggregate MAE**. It is not a prediction interval for an individual patient's age and does not include variation from retraining with different seeds.

### Reading the Outputs

- **MAE:** Average absolute difference between predicted and actual age.
- **MSE:** Average squared error; gives greater weight to large errors.
- **RMSE:** Square root of MSE, expressed in years.
- **R²:** Performance relative to the variation in the true ages; it is not classification accuracy.
- **Bias:** Mean of `prediction − actual`.
- **Residual plots:** Reveal systematic overestimation or underestimation.
- **Small/large error examples:** Illustrate selected cases, not a random sample of typical performance.

Since this is regression, classification accuracy, confusion matrices, and false-positive/false-negative counts do not replace the regression metrics.

---

## 11. Saved Outputs and Reproducibility

Each run creates a separate directory under `nih_runs`.

| File | Contents |
|---|---|
| `config.json` | Configuration and environment information |
| `train_split.csv` | Training manifest |
| `val_split.csv` | Validation manifest |
| `test_split.csv` | Test manifest |
| `data_audit.csv` | Data validation summary |
| `E1_history.csv` … `E4_history.csv` | Training histories |
| `E1_best.pth` … `E4_best.pth` | Best checkpoints |
| `validation_results.csv` | Validation results |
| `test_results.csv` | Test results |
| `E1_predictions.csv` … `E4_predictions.csv` | Predictions and errors |
| `figures/` | Saved visualizations |
| `summary_vi.md` | Automatically generated Vietnamese result summary |

Keep the configuration, splits, checkpoints, and metrics from the same run together.

Do not combine a checkpoint from one run with metrics from another.

A fixed seed supports reproducibility but does not guarantee identical results across all hardware and software environments.

---

## 12. GitHub Collaboration Workflow

### Contribution Process

1. Define the task in an Issue or `01_MEMBER_TASKS.md`.
2. Create a dedicated branch.
3. Update the relevant code or documentation.
4. Verify the modified cells and associated outputs.
5. Commit only the necessary files.
6. Open a Pull Request.
7. Request a teammate's review before merging.

### Example: Updating the README

Run from the repository root after saving or committing any existing work:

```bash
git switch main
git pull --ff-only
git switch -c docs/coursework-readme

git diff -- CourseWork/README.md
git add CourseWork/README.md
git commit -m "docs: update regression coursework README"

git push -u origin docs/coursework-readme
```

Then open a Pull Request from the new branch into `main`.

### Repository Practices

- Assign one person to integrate the final notebook.
- Preserve genuine outputs in the submission notebook.
- Record configuration changes when modifying experiments.
- Download the dataset separately instead of committing all images and ZIP archives.
- Record actual contributions in `01_MEMBER_TASKS.md`.
- Verify AI-assisted explanations and numerical claims against real notebook outputs.

### Suggested Responsibility Areas

| Area | Expected deliverables |
|---|---|
| Coordination | Project scope, schedule, and final integration |
| Data preparation | Metadata validation, audit, and patient-level splits |
| Model development | CNN + GAP + Linear and shape verification |
| Training and losses | E1–E4, checkpoint selection, and learning curves |
| Augmentation | Transform implementation and controlled comparisons |
| Evaluation and reporting | Metrics, error analysis, README, report, and slides |

Actual member assignments and contributions should be documented separately.

---

## 13. Troubleshooting

| Problem | Suggested action |
|---|---|
| `sample_labels.csv` not found | Extract the dataset and check `DATA_DIR` |
| CSV does not contain `Patient Age` | Use metadata that includes the age target |
| Repeated `CourseWork/CourseWork` path | Check `Path.cwd()` or use an absolute path |
| Missing Python package | Install it in the notebook kernel's environment |
| GPU is unavailable | Check the PyTorch installation and `torch.cuda.is_available()` |
| Training takes too long before presentation | Review saved outputs or use `reload` |
| Reload reports a dataset mismatch | Verify the metadata, images, and manifests belong to the same run |
| Checkpoint is missing | Check `RUN_DIR` and the `<experiment>_best.pth` file |

---

## 14. Limitations and Future Work

### Limitations

- The experiments use a random sample rather than the complete NIH dataset.
- Results are based on one seed and one patient split.
- Ten epochs do not establish convergence or optimal tuning.
- Extreme age groups have fewer observations.
- Large individual prediction errors remain.
- Patient and exact-image separation do not eliminate every possible source of dataset bias.

### Future Work

- Repeat experiments with multiple random seeds.
- Assess the stability of the experiment ranking.
- Tune hyperparameters using validation data.
- Improve representation of underrepresented age groups.
- Evaluate the model on an independent dataset.
- Investigate persistent age-dependent bias.

---

## 15. Conclusion

The project implements a complete **chest X-ray age regression pipeline** using CNN + Global Average Pooling + Linear output.

Four experiments compare MAE and MSE loss with and without augmentation.

In the final run, **E3 — MSE with augmentation** achieves a test MAE of **11.432 years**, RMSE of **14.211 years**, and R² of **0.316**, reducing MAE by approximately **19%** relative to the training-mean baseline.

However, errors remain substantial at the extremes of the age distribution. The model is an academic coursework experiment and requires further validation before any clinical application.

---

## 16. References

- [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- Final notebook: `NIH_Age_Regression_Nhom3_Local.ipynb`.
- Reported run: `full_20260925_191518_261037`.

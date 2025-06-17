# CS 4375 — Linear Regression Assignment

Two end–to–end solutions for predicting **Wine Quality** are provided:

| Part | File      | Approach                              |
|------|-----------|---------------------------------------|
| 1    | `part1.py`| **From-scratch** batch–gradient descent|
| 2    | `part2.py`| **scikit-learn** (`SGDRegressor`, `LinearRegression`)|

If the graders cannot execute your code exactly as described below **you will receive no credit**, so please follow the build/run steps verbatim when testing locally.

---

## 1.  Quick Start

```bash
# 1) Clone or download the repo
git clone <repo-url>
cd assignment                                 # <- make sure you are inside the folder

# 2) (Optional but recommended) create an isolated environment
python -m venv venv
source venv/bin/activate                     # Windows: venv\Scripts\activate

# 3) Install ALL dependencies in one command
pip install -r requirements.txt
# --- or, if you prefer ---
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo warnings
### Run Part 1 (custom GD)

```bash
python part1.py
```

### Run Part 2 (scikit-learn)

```bash
python part2.py
```

Both scripts download the dataset automatically, train the models, print metrics to the console and write logs/plots to the project folder.

---

## 2.  Repository Layout

```
assignment/
├── part1.py                           # custom gradient descent
├── part2.py                           # scikit-learn implementation
├── requirements.txt                   # exact package list (used by pip)
├── README.md                          # you are here
# ↓ generated after running the scripts ↓
├── hyperparameter_tuning_log.csv
├── sgd_hyperparameter_tuning_log.csv
├── regression_results.png
└── ml_library_regression_results.png
```

---

## 3.  Libraries Used

### 3.1  Common to **both** parts
* numpy
* pandas
* matplotlib
* seaborn
* ucimlrepo (automatic dataset download)
* warnings (suppresses benign runtime warnings)

### 3.2  **Part 2** specific
* scikit-learn ≥ 0.24  
  • `SGDRegressor` – stochastic gradient descent  
  • `LinearRegression` – closed-form ordinary least squares  
  • `StandardScaler`, `train_test_split`, `mean_squared_error`, etc.

No other third-party libraries are required.

---

## 4.  Dataset

* **Name:** Wine Quality  
* **Source:** UCI Machine Learning Repository  
* Fetched automatically via `ucimlrepo`; there is **no manual download step**.
* ≈ 6 000 samples, 11 numerical features, 1 quality score target (0–10).

---

## 5.  Expected Outputs

After each run you will see:

* Console summary: data shape, hyper-parameter search progress, evaluation metrics (MSE, RMSE, MAE, R², etc.).
* CSV log files with every hyper-parameter trial.
* PNG figures (`regression_results.png`, `ml_library_regression_results.png`) containing
  * cost-history plot
  * predicted-vs-actual scatter
  * residual analysis
  * feature-importance bar chart.

Typical runtimes on a modern laptop:  
* **Part 1:** 2–3 min  
* **Part 2:** 5–7 min (larger grid search)

---

## 6.  Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: ucimlrepo` | `pip install ucimlrepo` |
| Non-UTF-8 characters in console | Ensure your terminal is using UTF-8 encoding |
| Slow execution / high memory | Reduce the hyper-parameter grids inside the scripts |
| Plots not opening on some servers | They are still saved to PNG files in the folder |

If issues persist, delete any partially downloaded dataset in your home directory (`~/.ucimlrepo/`), reinstall the requirements, and rerun the script.

---

## 7.  Contact / Support

Team Members:  
John Hieu Nguyen - HMN220000  
Patrick Bui - PXB210047

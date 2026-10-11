# Detecting Faults on a Three-Phase Transmission Line

AI4003 Applied Machine Learning, Assignment 2 (Fall 2026)
Logistic regression written by hand in NumPy, and support vector machines (linear and RBF) with scikit-learn.

**Team:** `<Member 1 name, roll no.>` and `<Member 2 name, roll no.>`
**Blog:** `<Medium link>`  **LinkedIn post:** `<LinkedIn link>`

## The problem

A protective relay on a transmission line must decide within milliseconds whether the line has a fault. The input is one instant of the three line currents (Ia, Ib, Ic, in amperes) and the three line voltages (Va, Vb, Vc, in per unit). The output is 1 (fault) or 0 (no fault).

The question we answer: can a straight-line decision boundary detect a fault from raw instantaneous measurements? If not, is the simplest fix new features or a different classifier?

**Dataset:** [Electrical Fault Detection and Classification](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification) (Kaggle), file `detect_dataset.csv`, 12,001 rows. The two empty columns (`Unnamed: 7`, `Unnamed: 8`) are removed and the label `Output (S)` is renamed `fault`.

## Method in short

- **Split:** 60 / 20 / 20 train / validation / test with `np.random.default_rng(42)` (7200 / 2400 / 2401 rows).
- **Standardisation:** mean and standard deviation come from the training rows only and are applied to all three splits.
- **Set R:** the 6 raw measurements. **Set Q:** the 6 raw measurements plus their 6 squares (squares computed from raw values first, then all 12 columns standardised).
- **Logistic regression** (NumPy): sigmoid, log-likelihood cost with clipping, batch gradient descent, learning rate chosen as the largest alpha for which J decreases at every iteration.
- **SVMs** (`sklearn.svm.SVC`, Set R only): linear with C in {0.1, 1, 10, 100}, RBF with C in {0.1, 1, 10, 100} and gamma in {0.1, 1}. Hyper-parameters are chosen on the validation set.
- **Metrics** (`accuracy`, `confusion_matrix`) are written by hand. The test set is used once.

Allowed libraries only: numpy, pandas, matplotlib, joblib, Streamlit, and `sklearn.svm.SVC`.

## Results

**Test set (2401 rows), evaluated once**

| Model | Accuracy | False alarms | Missed faults |
|---|---|---|---|
| Logistic regression, Set R | 72.64% | 0 | 657 |
| Linear SVM (C = 0.1) | 73.59% | 0 | 634 |
| Logistic regression, Set Q | 97.88% | 6 | 45 |
| RBF SVM (C = 100, gamma = 1) | 99.58% | 4 | 6 |

**Logistic regression settings:** Set R used alpha = 1 (validation accuracy 73.08%), Set Q used alpha = 10 (validation accuracy 97.83%), both with 2000 iterations.

**Main finding:** raw instantaneous measurements are not linearly separable, so both straight-line models stay near 73%. A fault raises the current amplitude, but the sign of an instantaneous value alternates, so large positive and large negative currents are both faults. Adding squared features lets the same logistic regression reach 97.9%, and an RBF kernel reaches 99.6% without changing the features.

All tables and figures are in `results/`. The written answers to the Task 6 questions are in [`results/analysis.md`](results/analysis.md).

## Repository layout

```
data/
  detect_dataset.csv        (download from Kaggle, not committed if large)
src/
  load_data.py              the given loader
  prepare.py                Task 1: cleaning, split, standardisation
  eda.py                    Task 2: exploratory plots
  logistic.py               sigmoid, hypothesis, cost, fit_logistic, predict
  metrics.py                accuracy, confusion_matrix
  train_logistic.py         Task 3
  train_svm.py              Task 4
  test_models.py            Task 5
results/                    tables, figures, analysis.md, saved models and theta
app/
  app.py                    Task 7: Streamlit front end
fault_detection_assignment2.ipynb   all tasks in one notebook
README.md
```

## How to run

1. **Install**
   ```bash
   pip install numpy pandas matplotlib scikit-learn joblib streamlit
   ```
2. **Get the data.** Download `detect_dataset.csv` from the Kaggle link above and place it in `data/`.
3. **Check the loader**
   ```bash
   python src/load_data.py
   ```
4. **Reproduce the results.** Either open the notebook and choose Restart kernel and Run All (the linear SVM with C = 100 takes about a minute), or run the scripts in this order from the project root:
   ```bash
   python src/prepare.py
   python src/eda.py
   python src/train_logistic.py
   python src/train_svm.py
   python src/test_models.py
   ```
   This writes the tables, figures, `theta.npz`, `preprocessing.npz` and the saved SVMs to `results/`.
5. **Start the app** (needs the files from step 4 in `results/`)
   ```bash
   streamlit run app/app.py
   ```
   Run it from the project root. If `data/detect_dataset.csv` is present, the app also lets you browse real rows.

## Front end

The app shows the hand-written Set Q logistic regression and the RBF SVM side by side. Move the sliders or type values for Ia, Ib, Ic, Va, Vb, Vc and both predictions update immediately. It also has a sweep plot (vary one measurement and watch each model's output), a dataset browser with the true label, and a bar chart of the Set Q weights.

| No-fault reading | Fault reading |
|---|---|
| ![No-fault](results/screenshot_no_fault.png) | ![Fault](results/screenshot_fault.png) |

![Sweep plot](results/screenshot_sweep.png)

## Correctness checks

- Empty columns and the label are not among the features.
- Standardisation statistics come from training rows only.
- J(theta) starts at 0.6931 (= log 2) when theta = 0.
- J(theta) decreases at every iteration with the chosen alpha.
- Each confusion matrix adds up to 2401 test rows.
- The test rows are never used to choose alpha, C or gamma.


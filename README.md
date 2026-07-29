# Event Attendance Prediction: Bay Area, San Francisco

A machine learning pipeline to predict ticket sell-through (% Tickets Sold) for Bay Area cultural and community events, using ensemble methods with domain-specific optimization techniques.

---

## Problem Statement

Given publicly observable event characteristics — timing, location, category, and scale, predict what percentage of available tickets an event will sell. The practical use case is helping event organizers set realistic capacity expectations before an event goes on sale.

---

## Dataset

- **Source:** Bay Area Indian-American (and mixed) cultural event attendance records, 2000–2025
- **Size:** 1664 rows
- **Events covered:** Indian Community Events (Diwali, Navratri, Holi, Garba, Patriotic), Tech Conferences (Google I/O, Apple WWDC, RSA Conference, Dreamforce), Cultural Festivals (SF Pride, Outside Lands, Chinese New Year Parade), and others.
- **Target variable:** `% Tickets Sold` — continuous, range 7.7% to 100.0%, mean ~74%
- **Train/Test split:** 80/20 stratified on quartile bins of target (`random_state=42`)
  - Train: 1,331 rows
  - Test: 333 rows

---

## Features

9 features total — 5 base features and 4 engineered interaction features.

### Base Features (5)

| Feature            | Source Column  | Encoding                                                                                |
| ------------------ | -------------- | --------------------------------------------------------------------------------------- |
| `Month_sin`        | Month          | sin(2π·month/12) — cyclical encoding                                                    |
| `Month_cos`        | Month          | cos(2π·month/12) — cyclical encoding                                                    |
| `EventCat_encoded` | Event Category | Target encoding — mean % sold per category, smoothing=10, computed on training set only |
| `Location_encoded` | Location       | Target encoding — mean % sold per location, smoothing=10, computed on training set only |
| `Event_Scaled`     | Event Scale    | Ordinal — Small=0, Medium=1, Large=2                                                    |

### Interaction Features (4)

| Feature             | Computation                         | Rationale                                |
| ------------------- | ----------------------------------- | ---------------------------------------- |
| `Weekend_x_Holiday` | Is_Weekend_Day × Next_Day_Holiday   | Captures peak attendance conditions      |
| `Cat_x_Holiday`     | EventCat_encoded × Next_Day_Holiday | Category-specific holiday effect         |
| `Cat_x_Weekend`     | EventCat_encoded × Is_Weekend_Day   | Category-specific weekend effect         |
| `Scale_x_Cat`       | Event_Scaled × EventCat_encoded     | Scale interacts differently per category |

---

## Evaluation Metric

**±10pp Accuracy** — the primary metric used throughout this project.

A prediction is marked **correct (1)** if it falls within 10 percentage points of the actual value:

```
correct = 1  if  |predicted - actual| <= 10
correct = 0  otherwise

Accuracy = sum(correct) / total rows
```

This metric was chosen over standard regression metrics (R², MAE, RMSE) because it directly answers the practical question: is the model's prediction close enough to be operationally useful for capacity planning?

---

## Models

### Model 1 — XGBoost with Squared Epsilon Loss

Standard XGBoost gradient boosting with a **custom loss function** that directly aligns with the ±10pp evaluation metric.

**Loss function:**

```
L = max(0, (|actual - predicted| - 10)²)
```

This gives zero loss and zero gradient when the prediction is within ±10pp of the actual value, and a quadratic penalty for predictions outside the window. This is the squared epsilon-insensitive loss, implemented via XGBoost's custom objective API using explicit gradient and hessian computation.

**Results:**

| Set   | Correct | Total | Accuracy |
| ----- | ------- | ----- | -------- |
| Train | 1,154   | 1,331 | 86.70%   |
| Test  | 240     | 333   | 72.07%   |

---

### Model 2 — Random Forest with Sample Weighting

Bagging ensemble using Random Forest with **sample weights** to counteract systematic directional bias. The model historically overshoots low-attendance events (+31pp bias in 0–50% range) and undershoots high-attendance events (−16pp bias in 85–100% range). Sample weighting forces each tree's MSE-based splitting criterion to prioritise these harder ranges during training.

**Results:**

| Set   | Correct | Total | Accuracy |
| ----- | ------- | ----- | -------- |
| Train | 1,102   | 1,331 | 82.79%   |
| Test  | 256     | 333   | 76.88%   |

---

## Final Comparison

| Model                    | Train Accuracy | Test Accuracy |
| ------------------------ | -------------- | ------------- |
| XGBoost (Sq-ε loss)      | 86.70%         | 72.07%        |
| **RF (sample weighted)** | **82.79%**     | **76.88%**    |

Random Forest with sample weighting achieves the best test accuracy (76.88%) and the smallest train-test gap, indicating better generalisation. It is particularly strong in the 70–85% range (100% accuracy) and the 85–100% range (81.0%), where XGBoost struggles (60.0%).

XGBoost with the custom loss function shows higher training accuracy but a larger generalisation gap, reflecting the difficulty of applying gradient-based custom objectives to a dataset of this size (1,331 training rows).

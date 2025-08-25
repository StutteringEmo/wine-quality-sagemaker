# Wine Quality Prediction – AWS SageMaker

This project implements linear regression to predict wine quality (score 0–10) using the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

It was completed as part of **ANA680 – Applied Optimization (Problem 2)**, demonstrating ML deployment with and without container technology.

---

## Repository Structure

├── container/ # SageMaker with container technology
│ ├── WineQuality_Container.ipynb
│ ├── train.py
│ ├── inference.py
│
├── no_container/ # SageMaker without container technology
│ ├── WineQuality_NoContainer.ipynb
│ ├── wine_quality_lr_no_container.pkl
│
├── data/ # Training data
│ ├── winequality-red.csv
│ ├── winequality-white.csv
│ ├── winequality_combined.csv


---

## Problem 1: Heroku Deployment (with container)
- Built a Docker container with Linear Regression model
- Deployed to Heroku
- Submission includes **GitHub repo** and **Heroku URL**

---

##  Problem 2a: SageMaker without Container
- Trained Linear Regression using `SKLearn` inside SageMaker
- Saved model artifact (`wine_quality_lr_no_container.pkl`)
- No container tech used here

---

## Problem 2b: SageMaker with Container
- Created `train.py` and `inference.py`
- Used SageMaker SKLearn container
- Deployed model as a real-time endpoint
- Successfully tested predictions (CloudWatch logs checked)

---

## Results & Conclusion
- Both methods (with and without container) successfully trained Linear Regression models.
- The containerized approach required additional custom scripts (`train.py` + `inference.py`) but allowed flexible deployment as an endpoint.
- The no-container approach was simpler but less customizable.

---

## How to Reproduce
1. Upload dataset from `data/`
2. Run `WineQuality_NoContainer.ipynb` to train without container
3. Run `WineQuality_Container.ipynb` to train/deploy with container
4. Check CloudWatch logs if deployment fails

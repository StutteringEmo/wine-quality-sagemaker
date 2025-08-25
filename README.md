# Wine Quality Prediction – AWS SageMaker

This project predicts wine quality (a score between 0–10) using 11 features from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality):
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol

The objective was to explore two workflows in **AWS SageMaker**:

- **No Container**: Using SageMaker’s built-in SKLearn Estimator.
- **Container**: Using custom training (`train.py`) and inference (`inference.py`) scripts with the SKLearn container.

Both approaches train a **Linear Regression** model on the combined red and white wine datasets.

---

## Tech Stack
- **Language**: Python 3  
- **Libraries**: Scikit-learn, Pandas, Numpy, Joblib  
- **Environment**: JupyterLab on AWS SageMaker Studio  
- **Container**: AWS SageMaker SKLearn container  

---

## Repository Structure

- **container/** – SageMaker with container technology  
  - `WineQuality_Container.ipynb`  
  - `train.py`  
  - `inference.py`  

- **no_container/** – SageMaker without container technology  
  - `WineQuality_NoContainer.ipynb`  
  - `wine_quality_lr_no_container.pkl`  

- **data/** – Training data  
  - `winequality-red.csv`  
  - `winequality-white.csv`  
  - `winequality_combined.csv`  

---

## No Container Approach
- Implemented in `WineQuality_NoContainer.ipynb` using the built-in **SageMaker SKLearn Estimator**.  
- Model is trained and evaluated directly in the notebook.  
- Trained model artifact is saved as `wine_quality_lr_no_container.pkl`.  
- This method is quick to set up but has limited flexibility for customization.  

---

## Container Approach
- Implemented in `WineQuality_Container.ipynb`.  
- Uses custom scripts:  
  - `train.py` for training  
  - `inference.py` for deployment logic  
- Built on the AWS SageMaker **SKLearn container**.  
- Model is deployed as a real-time endpoint on SageMaker.  
- A smoke test was run to confirm predictions from the live endpoint.  
- This method provides full control and scalability for real-world deployment.  

---

## Results & Conclusion
- Both approaches successfully trained a Linear Regression model to predict wine quality.  
- **No Container**: simpler and faster, good for quick experiments.  
- **Container**: more setup required, but offers greater flexibility and production readiness.  
- Example endpoint prediction: **4.9976** (wine quality score).  

---

## How to Run
1. Upload the dataset files from `/data` into SageMaker Studio.  
2. Run `WineQuality_NoContainer.ipynb` for the no-container version.  
3. Run `WineQuality_Container.ipynb` for the containerized version.  
   - Ensure `train.py` and `inference.py` are in the same directory.  
4. Confirm the deployed endpoint by running the smoke test provided in the notebook.  

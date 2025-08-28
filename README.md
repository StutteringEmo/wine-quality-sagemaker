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
- **Container**: Training & inference scripts with SageMaker SKLearn container.

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

## How to Run
1. Upload the dataset files from `/data` into SageMaker Studio.  
2. Run `WineQuality_NoContainer.ipynb` for the no-container version.  
3. Run `WineQuality_Container.ipynb` for the containerized version.  
   - Ensure `train.py` and `inference.py` are in the same directory.  
4. Confirm the deployed endpoint by running the smoke test provided in the notebook.

---

## How to Run (Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/StutteringEmo/wine-quality-sagemaker.git
   cd wine-quality-sagemaker
   ```
   
2. Build the container locally (optional, since image is also on GHCR):
   ```bash
   docker build -t wine-quality-sagemaker ./container
   ```

3. Or pull the prebuilt image from GHCR:
   ```bash
   docker pull ghcr.io/stutteringemo/wine-quality-sagemaker:latest
   ```

4. Run the container:
   ```bash
   docker run --rm -p 8000:8000 ghcr.io/stutteringemo/wine-quality-sagemaker:latest
   ```

5. Open the Swagger docs to test the API:
   - Navigate to: http://localhost:8000/docs

<img width="914" height="569" alt="image" src="https://github.com/user-attachments/assets/6df2388a-da28-4e69-a5b8-d672966b399a" />

<img width="865" height="476" alt="image" src="https://github.com/user-attachments/assets/14f30443-e799-4792-8272-bf31811da6be" />

6. Try the `/predict` endpoint by providing JSON input like:
<img width="878" height="617" alt="image" src="https://github.com/user-attachments/assets/cd7f744a-74ed-4b15-8e08-6befc9372b33" />

Example Response:
<img width="855" height="620" alt="image" src="https://github.com/user-attachments/assets/eb8f9e66-9f0a-42b3-b8e3-901fb05924bb" />

---

## Results & Conclusion
- Both approaches successfully trained a Linear Regression model to predict wine quality.  
- **No Container**: simpler and faster, good for quick experiments.  
- **Container**: more setup required, but offers greater flexibility and production readiness.  
- Example endpoint prediction: **4.9976** (wine quality score).  

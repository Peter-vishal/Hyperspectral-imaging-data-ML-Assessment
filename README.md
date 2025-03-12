# Analysis of Hyperspectral Imaging Data for Mycotoxin Prediction

## Project Overview  
This project leverages hyperspectral imaging and machine learning to predict DON (Deoxynivalenol) mycotoxin levels in corn samples.  
The study follows a structured pipeline:  
- Data Preprocessing  
- Dimensionality Reduction (PCA & t-SNE)  
- Model Training & Evaluation (CNN & Transformer)  
- Interactive Predictions via Streamlit App  

---

## Dataset Description  
The dataset contains spectral reflectance values for corn samples across multiple wavelength bands.  
Key Attributes:  
- `hsi_id`: Identifier for each sample  
- `0 - 447`: Spectral band reflectance values  
- `vomitoxin_ppb`: DON concentration (target variable)  

---

## Technologies Used  
- Python (Pandas, NumPy, SciPy, Scikit-learn)  
- Deep Learning (TensorFlow/Keras, PyTorch)  
- Data Visualization (Matplotlib, Seaborn)  
- Dimensionality Reduction (PCA, t-SNE)  
- Deployment (Streamlit, Hugging Face Spaces)  

---

## Workflow  

### 1. Data Preprocessing
- Handled missing values with mean imputation  
- Removed outliers using IQR-based filtering  
- Standardized reflectance values using MinMaxScaler  
- Visualized spectral bands using line plots & heatmaps  

---

### 2. Dimensionality Reduction  
- PCA (Principal Component Analysis)  
   - Reduced data to 10 principal components  
   - First 3 components explained 85% of variance  

- t-SNE (t-Distributed Stochastic Neighbor Embedding)  
   - Visualized clusters based on DON levels  

---

### 3. Model Training  
- Implemented:  
   - CNN (Convolutional Neural Network)  
   - Transformer with Attention Mechanism  

- Hyperparameter Tuning (Random Search Optimization)  
   - CNN: Filters = [32, 64], Kernel Size = [5, 3]  
   - Transformer: 8 attention heads, feedforward dimension = 256  

---

### 4. Model Evaluation  
Performance Metrics (Lower is Better for MAE & RMSE)  

| Model         | MAE  | RMSE  | R² Score |
|--------------|------|------|----------|
| CNN          | 0.157 | 0.213 | 0.87     |
| Transformer  | 0.142 | 0.198 | 0.89     |

- Transformer outperformed CNN, showing better feature extraction  
- Scatter Plot: Actual vs. Predicted DON concentration  
- Residual Analysis: No major bias, errors are normally distributed  

---

### 5. Streamlit App for Predictions  
Features:  
- Upload spectral data (CSV format)  
- Predict DON concentration  
- Choose between CNN & Transformer models  
- Visualize scatter plots & confidence intervals  

Deployment:  
- Frontend: Streamlit  
- Backend: TensorFlow / PyTorch  
- Hosted on: Streamlit Cloud / Hugging Face Spaces  

---

## Key Findings & Future Improvements  
Findings:  
- Transformer-based models outperform CNNs for spectral data  
- PCA reduced dimensionality while preserving 85% variance  

Future Improvements:  
- Experiment with Graph Neural Networks (GNNs) for spectral-spatial relationships  
- Use data augmentation to improve generalization  
- Expand dataset with more diverse corn samples  

---

## Installation & Usage  
Clone the repository  
```bash
git clone (https://github.com/Peter-vishal/Hyperspectral-imaging-data-ML-Assessment)

# 🚨 Disaster Severity Prediction System

An end-to-end machine learning solution designed to predict the severity level of unfolding natural or structural disasters. This system processes raw environmental and tactical incident data through a robust ML pipeline and exposes a user-friendly interface via Streamlit for emergency responders and analysts.

---

## 📌 Problem Statement
In crisis management, rapid assessment is vital. When a disaster strikes, emergency dispatchers and rescue teams are flooded with fragmented data (e.g., rainfall metrics, affected radius, infrastructure types). Delaying severity categorization slows down resource allocation, which can cost lives. 

**The Goal:** Build an automated system that ingests incident metrics, processes them without data leakage, and outputs an instantaneous disaster severity prediction (Low, Medium, High) to accelerate critical decision-making.

---

## ⚙️ Core Approach & Architecture

This project transitions from exploratory code into a production-grade, modular architecture:

1. **Robust Preprocessing Pipeline (`src/preprocessing.py`)**: 
   * **Numerical Features**: Automatically handles missing values via median imputation and normalizes values using a standard scaler ($z$-score normalization).
   * **Categorical Features**: Handles missing categories with most-frequent imputation and vectorizes elements using One-Hot Encoding.
2. **Unified Pipeline Architecture (`src/train.py`)**: 
   * To completely prevent data leakage, the data preprocessor and the predictive model (e.g., Random Forest Classifier) are wrapped into a single Scikit-Learn `Pipeline` object. 
   * The *entire* integrated pipeline is exported as a unified binary artifact (`models/pipeline.pkl`).
3. **Decoupled Inference Engine (`src/predict.py`)**: 
   * Exposes a simple `ModelInference` wrapper class. The web app sends raw inputs, and the pipeline manages transformations and outputs seamlessly.
4. **Interactive UI App (`app/app.py`)**: 
   * Built with Streamlit to collect user inputs via forms, pass them through the backend inference engine, and visualize predictions along with confidence probability metrics.

---

## 📊 Results & Performance

* **Model Framework:** Random Forest Classifier / Tuned ML Model
* **Training Accuracy:** ~94.2% *(Update with your exact metrics)*
* **Testing Accuracy:** ~89.5% *(Update with your exact metrics)*

### Performance Summary
The unified pipeline successfully handles unseen categories during live inference without throwing runtime errors (`handle_unknown='ignore'`), providing stable and consistent runtime predictability.

---

## 🚀 How to Run the Project

Follow these steps to set up and run the pipeline and application on your local machine.

### 1. Prerequisites & Environment Setup
Clone the repository and navigate to the project directory. Create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install requirements safely using the python module flag
python -m pip install -r requirements.txt
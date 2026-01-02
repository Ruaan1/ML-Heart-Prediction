# Heart Disease Prediction Using Machine Learning

This project applies supervised machine learning techniques to predict the presence of heart disease using clinical and demographic patient data.

The focus is on **model design, preprocessing, and systematic hyperparameter tuning** rather than purely maximising performance. A feedforward neural network is used as the primary model, evaluated on a held-out test set.

---

## 📊 Dataset

The dataset is sourced from Kaggle and contains clinical measurements commonly used in cardiovascular risk assessment.

**Target variable**
- `HeartDisease`: binary label indicating presence (1) or absence (0) of heart disease

**Features include**
- Age, sex, cholesterol, resting blood pressure
- Chest pain type, ECG results, exercise-induced angina
- Maximum heart rate, ST depression, ST slope

The dataset is relatively clean and well-suited to structured/tabular classification tasks.

---

## 🧹 Preprocessing

Key preprocessing steps:
- Removal of missing and duplicate entries to maintain data integrity
- One-hot encoding of categorical features
- Standardisation of numerical features using `StandardScaler`
- Stratified splitting into training (60%), validation (20%), and test (20%) sets

This ensures balanced class representation and stable training behaviour.

---

## 🤖 Model Architecture

A **feedforward neural network (MLP)** was implemented using PyTorch.

**Architecture**
- Input layer: number of neurons equal to encoded feature count
- Hidden Layer 1: 16 neurons (ReLU)
- Dropout: 0.2
- Hidden Layer 2: 8 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Training configuration**
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Batch size: 64
- Learning rate: 0.001
- Epochs: 50

This architecture provides a strong baseline for tabular medical data without unnecessary complexity.

---

## 🔧 Hyperparameter Tuning

Extensive tuning was conducted to understand training dynamics:

- **Batch size**: evaluated from 1 → 80, with 64 yielding the most stable generalisation
- **Epoch count**: performance peaked around 50–60 epochs before overfitting emerged
- **Learning rate**: 0.001 produced the best balance between convergence speed and stability

Training and validation loss curves were analysed to detect overfitting and instability rather than relying on accuracy alone.

---

## 📈 Results

- **Test accuracy**: ~89% (performance range: 87–91%)
- Confusion matrix shows strong balance between true positives and true negatives
- Training and validation losses remain close, indicating good generalisation

The final model demonstrates that a relatively simple neural network can perform well on structured medical data when preprocessing and tuning are handled carefully.

---

## 🧠 Key Takeaways

- Proper preprocessing is as important as model choice
- Hyperparameter tuning should be guided by validation behaviour, not just accuracy
- Neural networks can perform competitively on tabular data when regularised correctly
- Performance gains beyond this point likely require feature engineering or domain-specific signals rather than deeper architectures

---

## 👥 Collaboration

This project was completed collaboratively.

My contributions included:
- Data preprocessing and feature encoding
- Neural network implementation in PyTorch
- Hyperparameter tuning and evaluation
- Analysis and interpretation of results

Graphs, training curves, and additional model variants can be found in the `GRAPHS/` and `CODE/` directories.

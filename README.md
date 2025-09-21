# German Traffic Sign Classification 

This project was completed as part of COMP30027 – Machine Learning at the University of Melbourne (Semester 1, 2025).
The goal was to build supervised machine learning models to classify German traffic signs into 43 categories using both engineered features and deep learning approaches

--- 
## 📖 Project Overview

* Dataset: Subset of the GTSRB – German Traffic Sign Recognition Benchmark
* Task: Predict the correct traffic sign class given raw images and/or extracted features.
* Deliverables: Python code (Jupyter Notebook), report (PDF), and Kaggle competition predictions.
* Key Focus: Comparing classical ML models with feature engineering versus CNNs on raw image data.

---
## ⚙️ Models Implemented

Four major models were developed and compared:

1) Random Forest

  * Trained on engineered features.
  * Hyperparameters tuned via grid search + cross-validation.
  * Robust to irrelevant features and effective on non-linear data


2) Multi-Layer Perceptron (MLP)

  * Optimized with feature selection, ReLU activation, early stopping, and grid search over architecture and learning rate.
  * Performed strongly across validation splits

3) Stacking Ensemble

  * Combined Random Forest, MLP, and SVM.
  * Used class reweighting and meta-classifier tuning.
  * Achieved better performance than individual base learners

4) Convolutional Neural Network (CNN)
  * Built from raw images, progressively enhanced with:
  * Data augmentation
  * Deeper layers
  * Larger input resolution (96×96)
  * Adaptive learning rate + early stopping
Achieved 98.6% validation accuracy and 99.3% Kaggle test accuracy, outperforming all other models

---
## 📊 Results Summary
| Model                 | Validation Accuracy | Macro F1 |
| --------------------- | ------------------- | -------- |
| Random Forest (tuned) | 81.1%               | 78.2%    |
| MLP (tuned)           | 83.3%               | 80.3%    |
| Stacking Ensemble     | 87.9%               | 87.3%    |
| CNN (final, 96×96)    | **98.6%**           | —        |
* Best performer: CNN with high-resolution images.
* Key trade-off: Classical models are interpretable & efficient; CNNs demand higher computation but achieve near-human accuracy

---
## ▶️ Running the Project

All experiments are contained in the Jupyter notebook.
Execute cells sequentially to reproduce results:

1) Random Forest → Section 3.3.1
2) MLP → Section 3.3.2
3) Stacking Ensemble → Section 3.4
4) CNN → Section 3.5

Each section is self-contained and produces its own results, figures, and evaluation metrics.

---
## 🧩 Key Insights

* Feature engineering (LBP, Hu moments, PCA-reduced histograms) improved classical ML model performance and reduced dimensionality from 120 → 51 features
* Stacking ensembles enhanced generalization, but performance plateaued despite tuning.
* CNNs leveraged raw images effectively, learning visual distinctions that handcrafted features missed, particularly between visually similar speed-limit signs.
* Error analysis revealed persistent challenges with class imbalance and blurred/ambiguous signs.

---
## 📑 References
* Wolpert, D.H. (1992). Stacked Generalization. Neural Networks.
* Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS.
* GTSRB Dataset on Kaggle


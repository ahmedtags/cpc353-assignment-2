# CPC353 - Natural Language Processing (Assignment 2)

This repository contains the Jupyter Notebook implementation and PDF report for **CPC353: Natural Language Processing - Assignment 2** (Semester 1, Academic Session 2025/2026) at Universiti Sains Malaysia (USM).

## Course Details
- **Course Code:** CPC353
- **Course Name:** Natural Language Processing
- **Semester:** Semester 1, Year 3 (2025/2026)
- **Project Title:** Stock Trend Classification using Deep Learning (LSTM & Transformers)

---

## Assignment Overview

The objective of this assignment is to classify stock price movements (trends) using financial news sentiments and historical stock market data. The project involves:
1. **Data Pre-processing:** Reading financial headlines corpus, combining news data, and preprocessing text (handling tokenization, padding, truncation).
2. **LSTM Model with GloVe Embeddings:** Building and training an LSTM network using pre-trained GloVe word vectors to classify news text sentiment.
3. **Transformer Model:** Implementing a Transformer architecture (using custom self-attention mechanisms or pre-trained models) to compare performance against the LSTM approach.
4. **Staging & Evaluation:** Comparing training times, classification metrics (accuracy, precision, recall, F1-score), and analyzing model behaviors.

---

## What I Did
- Wrote [`Part1b_Data_Preparation FINAL.ipynb`](Part1b_Data_Preparation%20FINAL.ipynb) to clean and align financial text with market movements.
- Built and evaluated the LSTM network in [`Part2_LSTM_GloVe FINAL.ipynb`](Part2_LSTM_GloVe%20FINAL.ipynb).
- Built and evaluated the Transformer architecture in [`Part3_Transformer FINAL.ipynb`](Part3_Transformer%20FINAL.ipynb).
- Documented model comparative results, analysis, and conclusions in the final report: [`Final_Report.pdf`](Final_Report.pdf).
- Utilized dataset `stock_trend.csv` containing aligned textual headlines and classification targets.

---

## Tools & Tech Stack
- **Language:** Python (Jupyter Notebook)
- **Frameworks/Libraries:** PyTorch / TensorFlow, NLTK, Scikit-learn, Pandas, Matplotlib, GloVe Pre-trained Embeddings

---

## How to Run

1. Open Jupyter notebook or Google Colab.
2. Ensure you have the `stock_trend.csv` dataset in the same directory.
3. Run the data preparation script: `Part1b_Data_Preparation FINAL.ipynb`.
4. Run `Part2_LSTM_GloVe FINAL.ipynb` to train/evaluate the LSTM model.
4. Run `Part3_Transformer FINAL.ipynb` to train/evaluate the Transformer model.

---

## 📸 Sample Output

Here is the summary of the training run and test-set performance metrics for the deep learning models:

### 1. Part 2: LSTM Model (GloVe-50 Embeddings)

#### Model Setup:
- **Embedding:** `glove-wiki-gigaword-50` (50 dimensions)
- **Architecture:** LSTM (64 units) with return sequences, Dense output layer with softmax activation.
- **Out of Vocabulary (OOV) Rate:** 11.36%

#### Training Progress (20 Epochs):
```
Epoch 1/20 - accuracy: 0.5396 - loss: 1.0713 - val_accuracy: 0.3713 - val_loss: 1.1306
Epoch 5/20 - accuracy: 0.5886 - loss: 0.8247 - val_accuracy: 0.5068 - val_loss: 1.0033
Epoch 10/20 - accuracy: 0.6481 - loss: 0.5982 - val_accuracy: 0.5510 - val_loss: 0.9421
Epoch 20/20 - accuracy: 0.6974 - loss: 0.3456 - val_accuracy: 0.6120 - val_loss: 0.9103
```

#### Test Set Evaluation (Classification Report):
```
Accuracy:  0.6570
Precision: 0.3676 (macro avg)
Recall:    0.4563 (macro avg)
F1 Score:  0.3450 (macro avg)

              precision    recall  f1-score   support
   downtrend       0.04      0.28      0.07        53
        flat       0.96      0.68      0.79      2261
     uptrend       0.11      0.41      0.17       123
```

---

### 2. Part 3: Transformer Model (Fine-tuned BERT)

#### Model Setup:
- **Base Model:** `bert-base-uncased` (109.5M parameters)
- **Hyperparameters:** Epochs = 3, Batch Size = 16, LR = 2e-5, Max Seq Length = 128

#### Fine-Tuning Run:
```
Epoch 1/3 - Train Loss: 1.0366, Train Acc: 0.8530 - Val Loss: 1.0194, Val Acc: 0.8289
Epoch 2/3 - Train Loss: 0.9961, Train Acc: 0.8758 - Val Loss: 1.3483, Val Acc: 0.9221 (Saved!)
Epoch 3/3 - Train Loss: 0.8604, Train Acc: 0.9046 - Val Loss: 1.1850, Val Acc: 0.8824
```

#### Test Set Evaluation (Classification Report):
```
Accuracy:  0.9224
Precision: 0.5204 (macro avg)
Recall:    0.3770 (macro avg)
F1 Score:  0.3960 (macro avg)

              precision    recall  f1-score   support
   downtrend       0.29      0.04      0.07        53
        flat       0.93      0.99      0.96      2261
     uptrend       0.34      0.11      0.16       123
```

*Conclusion: The fine-tuned BERT Transformer outperforms the GloVe-LSTM model significantly, raising test-set accuracy from 65.7% to 92.2% due to bidirectional contextual representations.*


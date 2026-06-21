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
5. Run `Part3_Transformer FINAL.ipynb` to train/evaluate the Transformer model.

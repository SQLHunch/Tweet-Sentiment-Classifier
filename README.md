# Sentiment140 Text-to-Sentiment Classification

## Problem Description

The goal of this project is to develop a machine learning model capable of analyzing tweets and classifying their sentiment into two categories: **positive** and **negative**. This is a classic **text classification task** in Natural Language Processing (NLP), where the input is the text of a tweet, and the output is the corresponding sentiment label.

### Significance of the Problem

Sentiment analysis is an essential tool in the modern world, enabling organizations to understand public opinion, emotions, and trends. It has a variety of applications, including:

- **Business and Marketing**:
  - Monitor customer satisfaction, product reviews, and feedback on social media platforms.
  - Evaluate the success of marketing campaigns and address customer concerns proactively.

- **Public Health and Social Good**:
  - Detect emotional distress, such as depression or anxiety, and alert healthcare professionals.
  - Gauge public sentiment about policies or crises, enabling data-driven decision-making.

- **Social Media Analytics**:
  - Understand public attitudes toward topics like elections, global events, or brands.
  - Identify trends and detect potentially harmful content.

### Dataset

This project uses the **Sentiment140 dataset**, which contains 1.6 million tweets labeled for sentiment. The dataset was automatically annotated using emoticons such as `:)` (positive) and `:(` (negative), reducing the need for manual labeling.

By solving this problem, the project demonstrates how machine learning can:

- Handle large-scale, real-world text data.
- Automate sentiment classification.
- Provide a foundation for further NLP applications like emotion detection or topic modeling.

---

## Model Architecture

The deep learning model used in this project is designed to effectively learn patterns in textual data for binary sentiment classification. Below are the key components:

### Embedding Layer

- **Purpose**: Converts tokenized input sequences into dense vector representations.
- **Parameters**:
  - Vocabulary size: 10,000 (most frequent words in the dataset).
  - Embedding dimensions: 128 (vector size for each token).

### Bidirectional LSTM Layers

- Two **Bidirectional LSTM (Long Short-Term Memory)** layers are used to capture temporal relationships in both forward and backward directions.
- **Layer 1**: 
  - Units: 128
  - Returns sequences to enable stacking of layers.
- **Layer 2**: 
  - Units: 64
  - Does not return sequences, providing a condensed output.
- **Benefit**: Enables the model to understand context from both past and future tokens.

### Dropout Layers

- Two **dropout layers** with a rate of 0.4 are added after each LSTM layer to reduce overfitting by randomly disabling neurons during training.

### Dense Layer

- A single neuron with a **sigmoid activation function** is used to output probabilities for binary classification (0 = Negative, 1 = Positive).

---

## Model Compilation

The model was compiled with the following parameters:

- **Optimizer**: Adam optimizer with a learning rate of `0.0001`, chosen for smooth convergence.
- **Loss Function**: Binary cross-entropy to measure the error for binary classification tasks.
- **Metrics**: Accuracy to monitor the proportion of correctly classified instances during training.

---

## Data Preparation

The text data was preprocessed and prepared as follows:

1. **Text Cleaning**: 
   - Removed URLs, mentions, hashtags, and special characters.
   - Converted text to lowercase.

2. **Tokenization and Padding**:
   - Converted text to sequences using a vocabulary size of 10,000.
   - Padded sequences to a maximum length of 100 tokens.

3. **Data Splitting**:
   - Split the dataset into **training (80%)** and **validation (20%)** sets to ensure robust evaluation.

---

## Training

The model was trained on the **training set** with the following parameters:

- **Epochs**: 5
- **Batch Size**: 64
- **Early Stopping**: Monitored validation loss to prevent overfitting by stopping training early if no improvement was observed.

The training process included:
- Tracking training and validation loss and accuracy.
- Saving the best model weights based on validation performance.

---

## Results

- **Validation Accuracy**: ~82%
- **Validation Loss**: ~0.39

### Learning Curves

The training and validation accuracy and loss were plotted to analyze model performance. The learning curves show consistent improvement over epochs, with convergence observed after the third epoch.

---

## Future Improvements

1. **Hyperparameter Tuning**: Experiment with different values for embedding dimensions, LSTM units, and learning rates.
2. **Data Augmentation**: Enhance the dataset with more diverse examples or techniques like synonym replacement.
3. **Explore Pre-trained Models**: Use transformer-based architectures like BERT for improved performance on textual data.
4. **Multi-class Sentiment**: Expand the model to classify tweets into positive, neutral, and negative sentiments.
5. **Transfer Learning**: Fine-tune the model on domain-specific datasets for better generalization.

---

## Getting Started

### Prerequisites

- Python 3.7 or above
- TensorFlow
- Required Python libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`

### Installation

Clone this repository:

```bash
git clone https://github.com/your-username/Sentiment140-Analysis.git
cd Sentiment140-Analysis


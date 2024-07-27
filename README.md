# SentimentAnalysisUsingSVM
This project is focused on performing sentiment analysis using a Support Vector Machine (SVM) classifier. The data used for training and testing the model is self-made and sourced from Kaggle. This README provides an overview of the project.
Sentiment analysis is a common task in Natural Language Processing (NLP) that involves determining the sentiment expressed in a piece of text. This project uses an SVM classifier to perform sentiment analysis on a dataset of text reviews of beauty product review in nepali language. The goal is to classify the sentiment of each review as positive or negative.
The dataset used in this project is self-made and available on Kaggle. It consists of text reviews labeled with their respective sentiments (positive or negative). The dataset can be downloaded from the following link:https://www.kaggle.com/datasets/yurikaprajapati/nepalibeautyreviewdataforsentimentanalysis?select=cleaned_nepali.csv

 SVM Process Description
1. Data Preprocessing
Loading Data: Load the dataset containing text reviews and their corresponding sentiment labels.
Text Cleaning: Remove any unwanted characters, symbols, or HTML tags from the text.
Tokenization: Split the text into individual tokens (words or phrases)using a suitable tokenizer for the Nepali languageor accordingly.
Stop Words Removal: Remove common words that do not carry significant meaning, such as "and", "the", etc. but in this nepali such as "र", "को", etc.
Stemming/Lemmatization: Reduce words to their base or root form to ensure consistency in the dataset.
2. Feature Extraction
Bag of Words (BoW): Convert the cleaned and tokenized text into a Bag of Words representation, where each text is represented as a vector of word counts.
TF-IDF Transformation: Apply Term Frequency-Inverse Document Frequency (TF-IDF) to the BoW representation to weigh the importance of words based on their frequency in the corpus.
3. Model Training
Pipeline Creation: Create a pipeline that sequentially applies the BoW, TF-IDF transformation, and SVM classifier to the data.
SVM Classifier: Configure the SVM classifier with appropriate kernel functions (e.g., linear, RBF) and hyperparameters.
Training the Model: Fit the SVM classifier to the training data to learn the relationship between the text features and their corresponding sentiment labels.
4. Model Evaluation
Performance Metrics: Evaluate the trained SVM model using metrics such as accuracy, precision, recall, and F1 score to assess its performance.
Confusion Matrix: Visualize the confusion matrix to understand the classification performance of the model in distinguishing between positive and negative sentiments.
5. Prediction
New Data Prediction: Use the trained SVM model to predict the sentiment of new, unseen text data.
Model Inference: Apply the entire preprocessing and feature extraction pipeline to the new data before making predictions with the SVM classifier.
This section provides a concise description of the SVM process for sentiment analysis, including data preprocessing, feature extraction, model training, evaluation, and prediction.


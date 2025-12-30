# Twitter_Sentiment_Analysis_project
Twitter Sentiment Analysis using Deep Learning & Transformers

Project Overview:
  This project focuses on building an end-to-end NLP sentiment analysis system to classify Twitter text as Positive or Negative.
Multiple deep learning approaches were implemented and compared, ranging from LSTM-based models to a Transformer-based DistilBERT model, achieving strong real-world performance.

Objectives:
   Perform sentiment classification on noisy social media text

   Apply NLP preprocessing and tokenization techniques

   Compare RNN-based models with Transformer architectures

   Fine-tune a pretrained Transformer for improved accuracy
   
Models Implemented:
LSTM  == Sequence-based model capturing contextual dependencies
Bidirectional LSTM  == Captures past and future context
DistilBERT (Transformer)  ==  Pretrained language model fine-tuned for sentiment classification

Tech Stack:
   Programming Language: Python
   
Libraries:
   Pandas, NumPy
   NLTK
   TensorFlow / Keras
   PyTorch
   HuggingFace Transformers
   Scikit-learn
   
Dataset::
   Source: Sentiment140 Twitter Dataset
   
Size:
  ~400K tweets for LSTM/BiLSTM
  20K balanced samples for DistilBERT fine-tuning
  
Labels:
  0 → Negative
  1 → Positive   
  
Methodology:
  Data loading and exploration
  Text preprocessing (lowercasing, URL removal, stopwords, tokenization)
  Tokenization and padding for LSTM models
  Word embedding learning
  Model training and evaluation
  Transformer fine-tuning using attention masks and AdamW optimizer
  
Results::
  Model	     Validation Accuracy
  LSTM	        ~78%
  BiLSTM	      ~79%
  DistilBERT	  ~82%   
  
Key Insights::
Bidirectional LSTM improves contextual understanding over standard LSTM
Pretrained Transformers significantly outperform RNN-based models
DistilBERT converges quickly and generalizes well with minimal fine-tuning

Sample Inference::
predict_sentiment("I absolutely love this product!")
# Output: Positive 

Future Improvements::
Add neutral sentiment classification
Experiment with RoBERTa or BERT-large
Deploy model using FastAPI or Streamlit
Perform hyperparameter tuning

Conclusion::
 This project demonstrates a production-style NLP workflow, highlighting the advantages ofTransformer-based models over traditional RNN approaches for sentiment analysis tasks.

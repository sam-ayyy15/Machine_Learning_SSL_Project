import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file='data/synthetic_asian_hate_tweets.csv', output_dir='data'):
    """Preprocess the synthetic dataset for modeling"""
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Dataset loaded with {len(df)} samples")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert text to features using bag of words
    print("Converting text to features...")
    vectorizer = CountVectorizer(
        max_features=5000,  # Limit features to prevent overfitting
        min_df=2,           # Ignore terms that appear in less than 2 documents
        max_df=0.95,        # Ignore terms that appear in more than 95% of documents
        stop_words='english'  # Remove English stop words
    )
    
    # Fit and transform the text data
    X = vectorizer.fit_transform(df['text'])
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    
    # Save label mapping for reference
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:")
    print(label_mapping)
    
    # Save the preprocessed data
    print("Saving preprocessed data...")
    pickle.dump(X, open(f"{output_dir}/X_features.pkl", 'wb'))
    pickle.dump(y, open(f"{output_dir}/y_labels.pkl", 'wb'))
    pickle.dump(vectorizer, open(f"{output_dir}/count_vectorizer.pkl", 'wb'))
    pickle.dump(label_mapping, open(f"{output_dir}/label_mapping.pkl", 'wb'))
    
    print(f"Preprocessing complete. Files saved to {output_dir}/")
    return X, y, vectorizer, label_mapping

if __name__ == "__main__":
    X, y, vectorizer, label_mapping = preprocess_data()
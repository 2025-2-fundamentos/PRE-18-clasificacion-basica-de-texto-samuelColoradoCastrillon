"""Text classification model training script."""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    # Load the data
    data = pd.read_csv(
        "files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(
        min_df=5,  # Minimum document frequency
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        stop_words='english',  # Remove common English stop words
        max_features=10000,  # Limit vocabulary size
    )
    
    # Transform the text data
    X = vectorizer.fit_transform(data.phrase)
    y = data.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the classifier
    clf = LogisticRegression(
        C=1.0,
        max_iter=200,
        class_weight='balanced',
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Save the vectorizer and classifier
    with open("homework/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("homework/clf.pickle", "wb") as f:
        pickle.dump(clf, f)

if __name__ == "__main__":
    main()
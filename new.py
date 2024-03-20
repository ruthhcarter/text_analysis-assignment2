# Step 1: Reading in the Data

import nltk
from nltk.corpus import twitter_samples

# Ensure twitter_samples is available
nltk.download('twitter_samples')

# Load the positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Verify the number of tweets loaded
print(f"Number of positive tweets: {len(positive_tweets)}")
print(f"Number of negative tweets: {len(negative_tweets)}")






# Step 2: Calculating Features

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer

# Initialize the tokenizer with settings to lower case and strip handles
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def tokenize_tweets(tweets):
    """Tokenize and join the tweets for vectorization."""
    return [" ".join(tokenizer.tokenize(tweet)) for tweet in tweets]

# Combine positive and negative tweets and tokenize
all_tweets = positive_tweets + negative_tweets
tokenized_tweets = tokenize_tweets(all_tweets)

# Initialize CountVectorizer to extract unigram and bigram features
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit and transform the tokenized tweets to get feature matrix
X = vectorizer.fit_transform(tokenized_tweets).toarray()

# Labels: 1 for positive, 0 for negative
y = [1] * len(positive_tweets) + [0] * len(negative_tweets)





# Step 3: Creating a Train/Test Set Split
from sklearn.model_selection import train_test_split

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")






# Step 4: Using the Linguistic Features to Make Predictions

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize and fit the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Optional: Identify the most predictive features
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
top_features = sorted(zip(coef, feature_names), reverse=True)[:10]
print("Top predictive features:", top_features)

#Import Libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Required downloads
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')


#Read CSV Files
df1 = pd.read_csv("mentalHealthAuthorsTS1.csv")
df2 = pd.read_csv("mentalHealthAuthorsTS2.csv")
df3 = pd.read_csv("giswTS2Authors.csv")
df4 = pd.read_csv("generalIssue.csv")
df5 = pd.read_csv("generalIssuesCommonTS1toTS2.csv")
df6 = pd.read_csv("generalIssueT1.csv")


#Check & Handle Missing Values
dfs = [df1, df2, df3, df4, df5, df6]
df_names = ["df1", "df2", "df3", "df4", "df5", "df6"]

print("Missing Values:\n")
for i, df in enumerate(dfs):
    print(f"{df_names[i]}:\n", df.isnull().sum(), "\n")
    df.fillna("", inplace=True)  # Missing values ko fill karna

print("Missing Values Removed Successfully")


#Check Categorical Columns
for i, df in enumerate(dfs):
    cat_cols = df.select_dtypes(include=['object']).columns
    print(f"\n{df_names[i]} Text Columns: {list(cat_cols)}")

    for col in cat_cols:
        print(f"Unique values in {col}: {df[col].nunique()}")


#Text Processing - Stopwords Removal, WordCloud, & Lemmatization
# Stopwords list
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Target column
target_col = 'selftext'

if target_col in df4.columns:
    text_data = " ".join(df4[target_col].dropna().astype(str))
    words = re.findall(r'\b\w+\b', text_data.lower())  
    words = [word for word in words if word not in stop_words]  

    word_freq = Counter(words)
    print("\nTop 10 Most Common Words in Selftext Column:")
    print(word_freq.most_common(10))

    # WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    # Lemmatization
    df4['selftext'] = df4['selftext'].astype(str).apply(lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))
    print("Lemmatization applied successfully on 'selftext'.")
else:
    print(f"{target_col} column not found in df4")


#Sentiment Analysis (VADER)
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

if 'selftext' in df4.columns:
    df4['sentiment'] = df4['selftext'].apply(get_sentiment)
    sentiment_counts = df4['sentiment'].value_counts()
    print("Sentiment Distribution:\n", sentiment_counts)

    # Plot sentiment distribution
    plt.figure(figsize=(6,4))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis Distribution")
    plt.show()


#Remove Duplicates
for i, df in enumerate(dfs):
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"{df_names[i]}: Removed {before - after} duplicate records.")


#TF-IDF Transformation
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

if 'selftext' in df4.columns:
    X_tfidf = vectorizer.fit_transform(df4['selftext'].astype(str))
    print("TF-IDF transformation completed.")


#Topic Modeling with LDA
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(X_tfidf)

def display_topics(model, feature_names, num_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))

display_topics(lda_model, vectorizer.get_feature_names_out(), 10)


#Train SVM Model for Sentiment Classification
if 'sentiment' in df4.columns:
    y = df4['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    print("SVM Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))



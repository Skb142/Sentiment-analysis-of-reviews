import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import os

# Load the dataset with a specified encoding
df = pd.read_csv(r'C:\Users\KIIT\Desktop\PROJECT\Phase1\ebay_reviews.csv', encoding='ISO-8859-1')

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Handle missing values by dropping rows with missing reviews
df.dropna(subset=['review'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize text in the review column
df['review'] = df['review'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Convert rating column to categorical
df['rating'] = df['rating'].astype('category')

# Sentiment analysis
df['sentiment'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative')

# Save the modified dataset
df.to_csv(r'C:\Users\KIIT\Desktop\PROJECT\Phase2\ebay_reviews_with_sentiment.csv', index=False)
print("Saved dataset as ebay_reviews_with_sentiment.csv")

# Ensure output directory exists
output_dir = r'C:\Users\KIIT\Desktop\PROJECT\Phase2\graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Saving plots to: {output_dir}")

# Distribution of ratings (1, 0, -1)
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df, hue='rating', palette='Set2', legend=False)
plt.title('Distribution of Ratings (1, 0, -1)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
print("Saved rating_distribution.png")
plt.show()
plt.close()

# Bar graph of positive and negative sentiments
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment_label', data=df, hue='sentiment_label', palette='Set2', legend=False)
plt.title('Distribution of Sentiment (Positive vs Negative)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
print("Saved sentiment_distribution.png")
plt.show()
plt.close()

# Word clouds for positive and negative reviews
positive_reviews = ' '.join(df[df['sentiment_label'] == 'Positive']['review'])
negative_reviews = ' '.join(df[df['sentiment_label'] == 'Negative']['review'])

if positive_reviews.strip():
    plt.figure(figsize=(10, 5))
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Reviews')
    plt.savefig(os.path.join(output_dir, 'wordcloud_positive.png'))
    print("Saved wordcloud_positive.png")
    plt.show()
    plt.close()

if negative_reviews.strip():
    plt.figure(figsize=(10, 5))
    wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Reviews')
    plt.savefig(os.path.join(output_dir, 'wordcloud_negative.png'))
    print("Saved wordcloud_negative.png")
    plt.show()
    plt.close()

# Calculate number of characters, words, and average word length
df['num_characters'] = df['review'].apply(len)
df['num_words'] = df['review'].apply(lambda x: len(x.split()))
df['avg_word_length'] = df['num_characters'] / df['num_words']

# Visualization of number of characters in reviews
fig, (pos_ax, neg_ax) = plt.subplots(1, 2, figsize=(15, 8))
len_pos_review = df[df['rating'] == 1]['review'].str.len()
pos_ax.hist(len_pos_review, color='green', bins=30)
pos_ax.set_title('Positive Reviews')
len_neg_review = df[df['rating'] == -1]['review'].str.len()
neg_ax.hist(len_neg_review, color='red', bins=30)
neg_ax.set_title('Negative Reviews')
fig.suptitle('Number of Characters in Reviews')
plt.savefig(os.path.join(output_dir, 'num_characters_reviews.png'))
print("Saved num_characters_reviews.png")
plt.show()
plt.close()

# Visualization of the number of words in reviews
fig, (pos_ax, neg_ax) = plt.subplots(1, 2, figsize=(15, 8))
pos_word = df[df['rating'] == 1]['review'].str.split().map(len)
pos_ax.hist(pos_word, color='green', bins=30)
pos_ax.set_title('Number of Words in Positive Reviews')
neg_word = df[df['rating'] == -1]['review'].str.split().map(len)
neg_ax.hist(neg_word, color='red', bins=30)
neg_ax.set_title('Number of Words in Negative Reviews')
fig.suptitle('Number of Words in Reviews')
plt.savefig(os.path.join(output_dir, 'num_words_reviews.png'))
print("Saved num_words_reviews.png")
plt.show()
plt.close()

# Visualize average word length in reviews
fig, (pos_ax, neg_ax) = plt.subplots(1, 2, figsize=(15, 8))
pos_word = df[df['rating'] == 1]['review'].str.split().apply(lambda x: [len(i) for i in x])
sns.histplot(pos_word.map(np.mean), ax=pos_ax, color='green', bins=30, kde=True)
pos_ax.set_title('Average Word Length in Positive Reviews')
neg_word = df[df['rating'] == -1]['review'].str.split().apply(lambda x: [len(i) for i in x])
sns.histplot(neg_word.map(np.mean), ax=neg_ax, color='red', bins=30, kde=True)
neg_ax.set_title('Average Word Length in Negative Reviews')
fig.suptitle('Average Word Length in Reviews')
plt.savefig(os.path.join(output_dir, 'avg_word_length_reviews.png'))
print("Saved avg_word_length_reviews.png")
plt.show()
plt.close()

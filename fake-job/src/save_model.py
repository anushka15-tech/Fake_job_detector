import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load cleaned data
df = pd.read_csv('data/processed.csv')
df['text'] = (
    df['title'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['requirements'].fillna('') + ' ' +
    df['benefits'].fillna('')
)
y = df['fraudulent']
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['text'])

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save trained model
with open('models/fake_job_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model & vectorizer saved!")

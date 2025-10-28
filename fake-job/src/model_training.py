import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# STEP 1: Load cleaned data
df = pd.read_csv('data/processed.csv')

# STEP 2: Feature engineering - combine all important text columns
df['text'] = (
    df['title'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['requirements'].fillna('') + ' ' +
    df['benefits'].fillna('')
)

# STEP 3: Define target
y = df['fraudulent']  # 1 = fake, 0 = genuine

# STEP 4: TF-IDF vectorizer (improved)
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['text'])

# STEP 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: SMOTE oversampling (handle imbalance)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# STEP 7: Train Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)

# STEP 8: Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

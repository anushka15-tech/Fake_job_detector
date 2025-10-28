import pickle

scam_keywords = [
    "immediate hiring", "no experience needed", "earn cash","form filling", "join now",
    "registration link", "daily payout", "form filling", "earn daily","referal bonus",
    "minimum work", "urgent hiring", "work from home", "instant cash",
    "instant payout", "limited seats", "urgent apply", "just internet","mega offer",
    "just fill forms", "earn 10000", "daily income","instant joining","cash transfer"
]

def contains_scam_keywords(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in scam_keywords)

with open('models/fake_job_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

title = "School Teacher"

company_profile = "Modern Public School, CBSE affiliated."

description = "Teach mathematics, prepare assignments, manage classroom."

requirements = "B.Ed, subject expertise, classroom experience"

benefits = "School transport, PF, staff quarters"


input_text = (
    title + " " +
    company_profile + " " +
    description + " " +
    requirements + " " +
    benefits
)

X_input = vectorizer.transform([input_text])

print("Scam keyword found?", contains_scam_keywords(input_text))
proba = model.predict_proba(X_input)[0][1]
print("Fake probability:", proba)

if contains_scam_keywords(input_text) or proba > 0.4:
    prediction = 1
else:
    prediction = 0

result_text = "Fake" if prediction == 1 else "Genuine"
print("Prediction:", result_text)

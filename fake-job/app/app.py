import streamlit as st
import pickle
import matplotlib.pyplot as plt
from db_utils import job_stats
from db_utils import create_table
from db_utils import insert_job
from db_utils import fetch_all_jobs

create_table()

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: #3366cc;'>🕵️‍♂️ Fake Job Posting Detector</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Predict whether a job posting is genuine or fake</h4>", 
    unsafe_allow_html=True
)
st.markdown("---")

# Scam keywords list (manual rules)
scam_keywords = [
    "immediate hiring", "no experience needed", "earn cash", 
    "registration link", "daily payout", "form filling", 
    "minimum work", "urgent hiring", "work from home", 
    "instant payout", "limited seats", "urgent apply",
    "just fill forms", "earn 10000", "daily income", "instant cash transfer"
]

def contains_scam_keywords(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in scam_keywords)

# Load model and vectorizer
with open('models/fake_job_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Form layout - organized
with st.form("job_post_form"):
    st.subheader("Enter Job Details")
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")
    submitted = st.form_submit_button("🔍 Predict Job Validity")

if submitted:
    # Simple check: koi field blank na ho
    if not title or not company_profile or not description or not requirements or not benefits:
        st.warning("❗ Please fill all the job details before submitting.", icon="⚠️")
    else:
        input_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements + " " +
            benefits
        )
        X_input = vectorizer.transform([input_text])
        proba = model.predict_proba(X_input)[0][1]  # Probability for 'Fake'

        # Manual + ML backend logic
        if contains_scam_keywords(input_text) or proba > 0.4:
            prediction = 1
        else:
            prediction = 0
        
        result = "Fake" if prediction == 1 else "Genuine"

        # Data save in DB (new line!)
        insert_job(title, company_profile, description, requirements, benefits, result)

        if prediction == 1:
            st.error("🚩 **Fake Job Posting Detected!**\nBe careful, this looks suspicious.", icon="🚩")
        else:
            st.success("✅ **Genuine Job Posting!**\nThis looks safe.", icon="✅")
        st.caption(f"Fake job probability: {proba:.2f}")


# Analysis
st.markdown("## 🔎 Job Prediction Analytics")

stats = job_stats()
if stats:
    labels = list(stats.keys())
    sizes = list(stats.values())

    # Dynamic colors mapping: Genuine = Green, Fake = Red, others = Grey
    colors = []
    for label in labels:
        label_clean = label.strip().lower()
        if label_clean == "genuine":
            colors.append("#4caf50")  # Green
        elif label_clean == "fake":
            colors.append("#f44336")  # Red
        else:
            colors.append("#cccccc")  # Grey for unknown/mistyped

    fig, ax = plt.subplots(figsize=(2.5,2.5))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140, colors=colors)
    ax.axis("equal")
    st.pyplot(fig)
    st.caption(f"Total jobs submitted: {sum(sizes)}")
else:
    st.info("No analytics data yet. Submit 1+ jobs to view stats.")



# dashboard
st.markdown("---")
if st.button("📋 Show All Submitted Jobs"):
    jobs = fetch_all_jobs()
    if jobs:
        import pandas as pd
        df = pd.DataFrame(jobs, columns=["Title", "Company", "Prediction"])
        df.index = df.index + 1
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No jobs submitted yet. Use the prediction form above to add jobs!")



# Footer
st.markdown(
    "<hr><center><small>Powered by Streamlit & Scikit-learn</small></center>", 
    unsafe_allow_html=True
)

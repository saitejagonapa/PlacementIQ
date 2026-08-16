import streamlit as st
import joblib
import pandas as pd

priority_features = [
    "AptitudeTestScore",
    "PlacementTraining",
    "ExtracurricularActivities",
    "SoftSkillsRating",
    "CGPA",
    "Projects",
    "Workshops/Certifications",
    "Internships"
]

# Thresholds selected using the median (50th percentile) of the training dataset for each numerical feature.
thresholds = {
    "AptitudeTestScore": 80,
    "SoftSkillsRating": 4.4,
    "CGPA": 7.7,
    "Projects": 2,
    "Workshops/Certifications": 1,
    "Internships": 1
}

def generate_suggestions(student, prediction):

    suggestions = []

    if prediction == 1:

        if student["Projects"] < thresholds["Projects"]:
            suggestions.append(
                "Build one or two advanced real-world projects to further strengthen your resume."
            )

        if student["Internships"] < thresholds["Internships"]:
            suggestions.append(
                "Gain additional internship experience to increase your practical industry exposure."
            )

        if student["Workshops/Certifications"] < thresholds["Workshops/Certifications"]:
            suggestions.append(
                "Complete more industry-recognized workshops or certifications to enhance your technical skills."
            )

        if student["AptitudeTestScore"] < thresholds["AptitudeTestScore"]:
            suggestions.append(
                "Continue practicing aptitude and logical reasoning to improve your performance in placement tests."
            )


    for feature in priority_features:

        if feature == "AptitudeTestScore":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Improve your aptitude score by practicing quantitative aptitude, logical reasoning, and verbal ability regularly."
                )

        elif feature == "PlacementTraining":
            if student[feature] == "No":
                suggestions.append(
                    "Participate in placement training programs to improve interview preparation and placement readiness."
                )

        elif feature == "ExtracurricularActivities":
            if student[feature] == "No":
                suggestions.append(
                    "Participate in extracurricular activities such as technical clubs, hackathons, volunteering, or leadership events."
                )

        elif feature == "SoftSkillsRating":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Improve your communication, teamwork, and interview skills through regular practice and mock interviews."
                )

        elif feature == "CGPA":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Improve your CGPA to strengthen your academic profile."
                )

        elif feature == "Projects":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Build more practical projects to strengthen your portfolio."
                )

        elif feature == "Workshops/Certifications":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Complete additional workshops or industry-recognized certifications to enhance your skills."
                )

        elif feature == "Internships":
            if student[feature] < thresholds[feature]:
                suggestions.append(
                    "Gain internship experience to improve your practical knowledge and industry exposure."
                )

    if len(suggestions) == 0:
        return [], True

    return suggestions[:3], False

@st.cache_resource
def load_model():
    model = joblib.load("model/logistic_regression_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, preprocessor, scaler

model, preprocessor, scaler = load_model()

st.set_page_config(
    page_title="PlacementIQ",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 PlacementIQ")
st.markdown(
    "<p style='font-size:20px; color:#A9A9A9;'>"
    "Machine Learning-Based Student Placement Prediction & Recommendation System"
    "</p>",
    unsafe_allow_html=True
)
st.header("👤 Student Profile")

st.divider()

st.subheader("📚 Academic Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA:",
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        value=7.0
    )
    
    ssc = st.number_input(
        "SSC Marks Percentage:",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        value=70.0
    )
    
    soft_skills = st.slider(
        "Soft Skills Rating:",
        min_value=3.0,
        max_value=4.8,
        value=4.0,
        step=0.1
    )

with col2:
    aptitude = st.number_input(
        "Aptitude Test Score:",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )
    
    hsc = st.number_input(
        "HSC Marks Percentage:",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        value=70.0
    )
    
    projects = st.number_input(
        "Number of Projects:",
        min_value=0,
        max_value=3,
        value=1
    )
    
st.divider()
    
st.subheader("💼 Professional Details")

col3, col4 = st.columns(2)

with col3:

    placement_training = st.selectbox(
        "Have you undergone placement training?",
        options=["Yes", "No"]
    )

    internships = st.number_input(
        "Number of Internships:",
        min_value=0,
        max_value=2,
        value=1
    )

with col4:

    workshops_certifications = st.number_input(
        "Workshops Attended/Certifications:",
        min_value=0,
        max_value=3,
        value=1
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities:",
        options=["Yes", "No"]
    )

predict = st.button("🔍 Predict Placement", use_container_width=True)

st.divider()

if predict:

    student_data = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Workshops/Certifications": [workshops_certifications],
        "AptitudeTestScore": [aptitude],
        "SoftSkillsRating": [soft_skills],
        "SSC_Marks": [ssc],
        "HSC_Marks": [hsc],
        "ExtracurricularActivities": [extracurricular],
        "PlacementTraining": [placement_training]
    })
    
    student_processed = preprocessor.transform(student_data)
    
    student_scaled = scaler.transform(student_processed)
    
    prediction = model.predict(student_scaled)[0]

    probability = model.predict_proba(student_scaled)[0][1]   
    
    if probability >= 0.75:
        confidence = "🟢 High"

    elif probability >= 0.40:
        confidence = "🟡 Medium"

    else:
        confidence = "🔴 Low"
    
    st.subheader("📊 Prediction Results")

    if prediction == 1:
        st.success(
    f"🎉 The model predicts that the student is likely to be placed with {probability*100:.1f}% confidence."
)
    else:
        st.error(
    "⚠️ The model predicts that the student currently has a lower chance of placement."
)
        
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Placement Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Confidence",
            confidence
        )

    st.divider()

    suggestions, excellent_profile = generate_suggestions(student_data.iloc[0], prediction)
        
    if excellent_profile:
        st.success(
            "🎉 Excellent profile! Continue practicing aptitude, coding, communication, and interview skills while applying to suitable companies."
        )
    else:
        if prediction == 1:
            st.success(
                "💡 The following suggestions can further strengthen your placement profile:"
            )
        else:
            st.warning(
                "🎯 Recommended Actions:"
            )
        for suggestion in suggestions:
            st.markdown(f"🔹 {suggestion}")

    
st.divider()

st.caption(
    "PlacementIQ • Built with Python, Streamlit and Scikit-learn"
)

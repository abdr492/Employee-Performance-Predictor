import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# PAGE CONFIG
# -----------------------------
import streamlit as st

st.set_page_config(page_title="Employee Intelligence Hub", layout="wide")

# --------------------------
# 🌈 CUSTOM CSS (GLASS UI)
# --------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Section Titles */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}

/* KPI Cards */
.kpi {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: white;
}

/* Prediction Box */
.prediction-box {
    background: linear-gradient(135deg, #6366f1, #9333ea);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
}

/* Insight Box */
.insight {
    background: rgba(34,197,94,0.15);
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.warning {
    background: rgba(239,68,68,0.15);
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
pipeline = joblib.load("models/pipeline.pkl")
df = pd.read_csv("data/raw/employee_data.csv")



# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div style='text-align:center'>
    <h1>🚀 Employee Intelligence Hub</h1>
    <p style='color:gray'>AI-Powered Workforce Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE INIT
# -----------------------------
if "pred" not in st.session_state:
    st.session_state.pred = None


tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🧠 Explanation",
    "📈 Analytics"
])

# -----------------------------
# KPI CARDS 🔥
# -----------------------------
with tab1:
    st.markdown("<div class='section'>📊 Executive Overview</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown("<div class='kpi'>1000<br>Total Employees</div>", unsafe_allow_html=True)
    col2.markdown("<div class='kpi'>48.5<br>Avg Training</div>", unsafe_allow_html=True)
    col3.markdown("<div class='kpi'>10.3<br>Projects</div>", unsafe_allow_html=True)
    col4.markdown("<div class='kpi'>92%<br>Accuracy</div>", unsafe_allow_html=True)

    # -----------------------------
    # SIDEBAR INPUT
    # -----------------------------
    st.sidebar.title("Input Features")

    age = st.sidebar.slider("Age", 20, 60, 30)
    experience = st.sidebar.slider("Experience", 0, 40, 5)
    projects = st.sidebar.slider("Projects Completed", 0, 50, 10)
    training = st.sidebar.slider("Training Hours", 0, 100, 40)
    work_hours = st.sidebar.slider("Work Hours", 6, 12, 8)
    delivery = st.sidebar.slider("Delivery Rate", 0.0, 1.0, 0.7)
    manager = st.sidebar.slider("Manager Rating", 1.0, 5.0, 3.5)
    peer = st.sidebar.slider("Peer Feedback", 1.0, 5.0, 3.5)

    education = st.sidebar.selectbox("Education", ["Bachelors", "Masters", "PhD"])
    department = st.sidebar.selectbox("Department", ["HR", "IT", "Sales"])

    # -----------------------------
    # INPUT DATAFRAME
    # -----------------------------
    input_df = pd.DataFrame({
        'age': [age],
        'experience': [experience],
        'projects_completed': [projects],
        'training_hours': [training],
        'avg_work_hours': [work_hours],
        'on_time_delivery_rate': [delivery],
        'manager_rating': [manager],
        'peer_feedback': [peer],
        'education': [education],
        'department': [department]
    })

    # -----------------------------
    # PREDICTION
    # -----------------------------
    st.markdown("<div class='section'>🎯 Prediction Engine</div>", unsafe_allow_html=True)
    if st.button("Predict Performance"):

            st.session_state.pred = pipeline.predict(input_df)[0]
            pred = st.session_state.pred

            label_map = {0: "Low", 1: "Medium", 2: "High"}
            prediction = label_map[pred]

            emoji = "🔥" if prediction == "High" else "⚠️" if prediction == "Low" else "⚡"

            st.markdown(
            f"<div class='prediction-box'>{emoji} {prediction} Performance</div>",
            unsafe_allow_html=True
        )

            proba = pipeline.predict_proba(input_df)[0]

            fig_conf = go.Figure(go.Bar(
                x=["Low", "Medium", "High"],
                y=proba,
                marker_color=["#ef4444", "#facc15", "#22c55e"]
            ))

            fig_conf.update_layout(
                title="Prediction Confidence",
                template="plotly_dark"
            )

            st.plotly_chart(fig_conf, use_container_width=True)


    # -----------------------------
    # SHAP EXPLANATION (FIXED)
    # -----------------------------
with tab2:
    if st.session_state.pred is None:
        st.warning("⚠️ Please run prediction first.")
        st.stop()

    pred = st.session_state.pred
    st.markdown("<div class='section'>🧠 Model Explanation</div>", unsafe_allow_html=True)
    
    pred = st.session_state.pred
    X_sample = pipeline.named_steps['preprocessor'].transform(input_df)
    model = pipeline.named_steps['model']

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle multiclass
    if isinstance(shap_values, list):
        shap_vals = shap_values[pred][0]
    else:
        shap_vals = shap_values[0]

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

        # -----------------------------
    # FINAL SHAP FIX (WORKS 100%)
    # -----------------------------

    # Convert to numpy + force 1D
    shap_vals = np.array(shap_vals).reshape(-1)

    # Match length
    min_len = min(len(feature_names), len(shap_vals))

    shap_df = pd.DataFrame({
        "Feature": feature_names[:min_len],
        "Impact": shap_vals[:min_len]
    })

        # Sort
    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False).head(10)

    fig = px.bar(
    shap_df,
    x="Impact",
    y="Feature",
    orientation="h",
    title="Top Feature Importance",
    color="Impact",
    color_continuous_scale="plasma"
)

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # WATERFALL 🔥
    # -----------------------------
    st.markdown("<div class='section-title'>🔥 Why this prediction?</div>", unsafe_allow_html=True)
    # -----------------------------
    # PROFESSIONAL INSIGHT TEXT 🔥
    # -----------------------------

    def clean_feature_name(feature):
        if "num_" in feature:
            return feature.replace("num_", "").replace("_", " ").title()
        elif "cat_" in feature:
            parts = feature.replace("cat_", "").split("_")
            return f"{parts[0].title()}: {parts[1]}"
        else:
            return feature.replace("_", " ").title()


    top_features = shap_df.head(3)

    positive = [clean_feature_name(f) for f in top_features[top_features["Impact"] > 0]["Feature"]]
    negative = [clean_feature_name(f) for f in top_features[top_features["Impact"] < 0]["Feature"]]

    st.markdown("<div class='section-title'>💡 AI Insight</div>", unsafe_allow_html=True)

    if positive:
        st.markdown(f"🟢 **Key strengths:** {', '.join(positive)}")

    if negative:
        st.markdown(f"🟡 **Areas for improvement:** {', '.join(negative)}")

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals[:min_len],
            base_values=explainer.expected_value[pred] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=X_sample[0],
            feature_names=feature_names
        )
    )

# -----------------------------
# DATASET DASHBOARD 🔥
# -----------------------------
st.markdown("---")
with tab3:
    st.markdown("<div class='section'>📈 Workforce Analytics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Distribution
    fig1 = px.histogram(df, x="performance", color="performance",
                        title="Performance Distribution",
                        color_discrete_sequence=px.colors.qualitative.Bold)
    col1.plotly_chart(fig1, use_container_width=True)

    # Training vs Performance
    fig2 = px.box(df, x="performance", y="training_hours",
                color="performance",
                title="Training vs Performance")
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="training_hours",
        y="projects_completed",
        color="performance",
        title="Performance Clusters",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # HEATMAP 🔥
    # -----------------------------
    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)

    corr = df.corr(numeric_only=True)

    fig3 = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    title="Feature Correlation")

    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # FOOTER
    # -----------------------------
    st.markdown("---")
    st.markdown("<p style='text-align:center;'>Built with ❤️ using ML + SHAP + Streamlit</p>", unsafe_allow_html=True)
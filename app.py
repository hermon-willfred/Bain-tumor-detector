import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px
import time

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_ai_model():
    model = load_model("brain_tumor_cnn.keras")
    return model

model = load_ai_model()

# ----------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.stApp{
background:#0f172a;
}

.title{
text-align:center;
font-size:50px;
font-weight:bold;
color:#00E5FF;
}

.subtitle{
text-align:center;
font-size:20px;
color:white;
}

.card{
background:#1e293b;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 15px cyan;
}

</style>
""",unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------

st.markdown("""
<div class='title'>
🧠 Brain Tumor Detection using CNN
</div>
""",unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Artificial Intelligence Based MRI Brain Tumor Classification System
</div>
""",unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📊 Statistics",
        "ℹ About"
    ]
)
# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.markdown("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        ## 👋 Welcome to Brain Tumor Detection AI

        This application uses **Convolutional Neural Networks (CNN)** to classify Brain MRI images into four categories.

        ### 🎯 Detectable Classes

        ✅ Glioma Tumor

        ✅ Meningioma Tumor

        ✅ Pituitary Tumor

        ✅ No Tumor

        Upload an MRI image in the Prediction page to get AI-based results.
        """)

    with col2:

        st.info("""
        ### 📌 Model Information

        • Algorithm : CNN

        • Framework : TensorFlow

        • Image Size : 224 × 224

        • Classes : 4

        • Output : Softmax
        """)

    st.write("")

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.subheader("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            label="📁 Training Images",
            value="5600"
        )

    with c2:

        st.metric(
            label="🧠 Classes",
            value="4"
        )

    with c3:

        st.metric(
            label="🤖 Model",
            value="CNN"
        )

    with c4:

        st.metric(
            label="📐 Image Size",
            value="224×224"
        )

    st.write("")

    # =====================================================
    # FEATURES
    # =====================================================

    st.subheader("✨ Features")

    left, right = st.columns(2)

    with left:

        st.success("""
        ✔ MRI Image Upload

        ✔ Automatic Prediction

        ✔ CNN Based Detection

        ✔ AI Confidence Score
        """)

    with right:

        st.info("""
        ✔ Probability Graph

        ✔ Medical Report

        ✔ Interactive Dashboard

        ✔ Fast Prediction
        """)

    st.write("")

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.subheader("⚙ How It Works")

    st.markdown("""
    ### Step 1
    📤 Upload Brain MRI Image

    ⬇

    ### Step 2
    🖼 Image Preprocessing

    ⬇

    ### Step 3
    🧠 CNN Model Prediction

    ⬇

    ### Step 4
    📊 Probability Calculation

    ⬇

    ### Step 5
    ✅ Final Diagnosis
    """)

    st.write("")

    # =====================================================
    # QUICK INFO
    # =====================================================

    st.subheader("📚 Brain Tumor Types")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Glioma",
        "Meningioma",
        "Pituitary",
        "No Tumor"
    ])

    with tab1:
        st.write("""
Glioma tumors originate from glial cells in the brain.
""")

    with tab2:
        st.write("""
Meningioma develops from the membranes covering the brain.
""")

    with tab3:
        st.write("""
Pituitary tumors develop in the pituitary gland.
""")

    with tab4:
        st.write("""
Normal MRI with no evidence of brain tumor.
""")

    st.write("")

    st.success("👉 Use the **Prediction** page from the sidebar to upload an MRI image and get an AI prediction.")
# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🔍 Prediction":

    st.markdown("---")
    st.header("🧠 Brain MRI Prediction")

    st.write("Upload a Brain MRI image to detect the tumor type using the trained CNN model.")

    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1,1])

        # =====================================================
        # DISPLAY IMAGE
        # =====================================================

        with col1:

            st.subheader("🖼 Uploaded MRI")

            st.image(
                image,
                use_container_width=True
            )

        # =====================================================
        # PREDICTION
        # =====================================================

        with col2:

            st.subheader("🤖 AI Prediction")

            if st.button("🔍 Predict Tumor", use_container_width=True):

                with st.spinner("Analyzing MRI Image..."):

                    image = Image.open(uploaded_file).convert("RGB")

                    image = image.resize((224,224))

                    img = np.array(image,dtype=np.float32)/255.0

                    img = np.expand_dims(img,0)

                    prediction = model.predict(img, verbose=0)

                    confidence = float(np.max(prediction)) * 100

                    predicted_class = classes[np.argmax(prediction)]

                    time.sleep(1)

                st.success("Prediction Completed Successfully ✅")

                st.metric(
                    "Predicted Class",
                    predicted_class
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                # ===============================================
                # MEDICAL MESSAGE
                # ===============================================

                if predicted_class == "No Tumor":

                    st.success("""
                    ### ✅ Result

                    No Brain Tumor Detected.

                    The MRI appears normal.
                    """)

                else:

                    st.error(f"""
                    ### ⚠ Detected

                    **{predicted_class}**

                    Please consult a neurologist for further diagnosis.
                    """)

                # ===============================================
                # PROBABILITY CHART
                # ===============================================

                st.subheader("📊 Prediction Probability")

                probability = pd.DataFrame({

                    "Tumor": classes,

                    "Probability": prediction[0] * 100

                })

                fig = px.bar(

                    probability,

                    x="Tumor",

                    y="Probability",

                    color="Tumor",

                    text_auto=".2f",

                    title="Class Probability"

                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ===============================================
                # DETAILED REPORT
                # ===============================================

                st.subheader("📋 AI Report")

                report = pd.DataFrame({

                    "Category": [
                        "Predicted Tumor",
                        "Confidence",
                        "Model",
                        "Image Size"
                    ],

                    "Result": [
                        predicted_class,
                        f"{confidence:.2f} %",
                        "CNN",
                        "224 x 224"
                    ]

                })

                st.table(report)

                # ===============================================
                # DOWNLOAD REPORT
                # ===============================================

                csv = report.to_csv(index=False)

                st.download_button(

                    "⬇ Download Report",

                    data=csv,

                    file_name="Brain_Tumor_Report.csv",

                    mime="text/csv"

                )

                # ===============================================
                # CLASS PROBABILITIES
                # ===============================================

                st.subheader("📈 Individual Class Confidence")

                for i in range(len(classes)):

                    st.progress(float(prediction[0][i]))

                    st.write(
                        f"**{classes[i]} : {prediction[0][i]*100:.2f}%**"
                    )

                st.balloons()
# =====================================================
# STATISTICS PAGE
# =====================================================

elif page == "📊 Statistics":

    st.markdown("---")
    st.header("📊 Model Statistics Dashboard")

    st.write("Overview of the CNN model and Brain MRI dataset.")

    # ============================================
    # MODEL METRICS
    # ============================================

    st.subheader("📈 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Training Images", "5600")

    with c2:
        st.metric("Testing Images", "1600")

    with c3:
        st.metric("Image Size", "224 × 224")

    with c4:
        st.metric("Classes", "4")

    st.markdown("---")

    # ============================================
    # DATASET DISTRIBUTION
    # ============================================

    st.subheader("🧠 Dataset Distribution")

    dataset = pd.DataFrame({

        "Class":[
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],

        "Images":[
            1626,
            1645,
            2000,
            1929
        ]

    })

    fig = px.pie(

        dataset,

        values="Images",

        names="Class",

        hole=0.45,

        title="Brain MRI Dataset"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================================
    # BAR CHART
    # ============================================

    st.subheader("📊 Images per Class")

    fig2 = px.bar(

        dataset,

        x="Class",

        y="Images",

        color="Class",

        text="Images"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ============================================
    # MODEL INFORMATION
    # ============================================

    st.subheader("🤖 CNN Model Information")

    info = pd.DataFrame({

        "Parameter":[

            "Model",

            "Framework",

            "Input Shape",

            "Optimizer",

            "Loss",

            "Activation",

            "Output Layer"

        ],

        "Value":[

            "CNN",

            "TensorFlow / Keras",

            "224×224×3",

            "Adam",

            "Categorical Crossentropy",

            "ReLU",

            "Softmax"

        ]

    })

    st.table(info)

    st.markdown("---")

    # ============================================
    # CLASS INFORMATION
    # ============================================

    st.subheader("🧠 Brain Tumor Classes")

    tumor = pd.DataFrame({

        "Class":[

            "Glioma",

            "Meningioma",

            "Pituitary",

            "No Tumor"

        ],

        "Description":[

            "Tumor arising from glial cells.",

            "Tumor originating from meninges.",

            "Tumor in pituitary gland.",

            "Healthy Brain MRI."

        ]

    })

    st.dataframe(
        tumor,
        use_container_width=True
    )

    st.markdown("---")

    # ============================================
    # WORKFLOW
    # ============================================

    st.subheader("⚙ AI Workflow")

    st.info("""

📤 Upload MRI

⬇

🖼 Image Preprocessing

⬇

🧠 CNN Feature Extraction

⬇

📊 Softmax Classification

⬇

✅ Final Prediction

""")

    st.markdown("---")

    # ============================================
    # PROJECT SUMMARY
    # ============================================

    st.subheader("📋 Project Summary")

    st.success("""

✔ Deep Learning Model : CNN

✔ MRI Image Classification

✔ 4 Brain Classes

✔ Automatic Prediction

✔ Confidence Score

✔ Probability Graph

✔ Download Report

✔ TensorFlow + Streamlit

""")
# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ About":

    st.markdown("---")

    st.header("ℹ About This Project")

    st.write("""

## 🧠 Brain Tumor Detection using Deep Learning

This project uses a **Convolutional Neural Network (CNN)** to classify Brain MRI images into four categories.

The application is developed using **TensorFlow**, **Keras**, **Python**, and **Streamlit** to provide an AI-powered interface for brain tumor detection.

""")

    st.markdown("---")

    # ============================================
    # PROJECT OBJECTIVE
    # ============================================

    st.subheader("🎯 Project Objective")

    st.success("""

• Detect brain tumors automatically using MRI images.

• Reduce manual diagnosis time.

• Improve prediction accuracy using CNN.

• Provide an easy-to-use web application.

""")

    st.markdown("---")

    # ============================================
    # TECHNOLOGIES
    # ============================================

    st.subheader("💻 Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.info("🐍 Python")

    with tech2:
        st.info("🤖 TensorFlow")

    with tech3:
        st.info("🧠 CNN")

    with tech4:
        st.info("🎈 Streamlit")

    st.markdown("---")

    # ============================================
    # MODEL DETAILS
    # ============================================

    st.subheader("⚙ Model Details")

    details = pd.DataFrame({

        "Parameter":[

            "Model",

            "Framework",

            "Input Size",

            "Output Classes",

            "Optimizer",

            "Loss Function"

        ],

        "Value":[

            "CNN",

            "TensorFlow/Keras",

            "224 × 224 × 3",

            "4",

            "Adam",

            "Categorical Crossentropy"

        ]

    })

    st.table(details)

    st.markdown("---")

    # ============================================
    # FEATURES
    # ============================================

    st.subheader("✨ Key Features")

    st.write("""

✅ Upload MRI Image

✅ Automatic Brain Tumor Detection

✅ Confidence Score

✅ Interactive Probability Chart

✅ Download Prediction Report

✅ Professional Dashboard

""")

    st.markdown("---")

    # ============================================
    # FUTURE SCOPE
    # ============================================

    st.subheader("🚀 Future Scope")

    st.write("""

• Deploy using Streamlit Cloud

• Doctor Login System

• Patient History

• PDF Medical Report

• Mobile Application

• Explainable AI (Grad-CAM)

• Multi-Disease Detection

""")

    st.markdown("---")

    # ============================================
    # DISCLAIMER
    # ============================================

    st.subheader("⚠ Medical Disclaimer")

    st.warning("""

This AI model is developed for educational and research purposes.

It should NOT replace a professional doctor's diagnosis.

Always consult a qualified neurologist or radiologist before making any medical decision.

""")

    st.markdown("---")

    # ============================================
    # DEVELOPER
    # ============================================

    st.subheader("👨‍💻 Developer")

    st.write("""

**Project Title**

Brain Tumor Detection using Deep Learning

**Developed Using**

- Python
- TensorFlow
- Keras
- CNN
- Streamlit
- Plotly

""")

    st.markdown("---")

    # ============================================
    # CONTACT
    # ============================================

    st.subheader("📧 Contact")

    st.info("""

Email : hermonwillfred39@gmail.com

College : Nehru Institute of technology

Department : Computer Science and engineering

""")

    st.markdown("---")

    # ============================================
    # FOOTER
    # ============================================

    st.markdown(
        """
        <center>

        <h3>🧠 Brain Tumor Detection AI</h3>

        <p>Developed using TensorFlow | CNN | Streamlit | Python</p>

        <p>© 2026 All Rights Reserved</p>

        </center>
        """,
        unsafe_allow_html=True
    )
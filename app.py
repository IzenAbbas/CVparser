import streamlit as st
import io
import base64
import re
import pickle
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    top_k_accuracy_score
)

# =========================== MODEL LOADING ===========================
# NOTE: These files must exist in the same directory as the script to run.
try:
    clf1 = pickle.load(open('clf1.pkl', 'rb'))
    clf2 = pickle.load(open('clf2.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('le.pkl', 'rb'))
    x_test = pickle.load(open('x_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files (clf1.pkl, clf2.pkl, etc.) not found. Please ensure they are in the correct directory.")
    st.stop()


# =========================== CLEANING FUNCTION ===========================
def clean_text(text):
    clean_txt = text
    # 2. Remove backlash since experience includes date in dd\mm\yy
    clean_txt = re.sub(r'\\', ' ', clean_txt)
    # 3. Remove common metadata/headers (RT, cc, etc.)
    clean_txt = re.sub(r'RT|cc|CC|rt', '', clean_txt)
    # 4. Remove URLs
    clean_txt = re.sub(r'http\S+', '', clean_txt)
    # 5. Remove mentions
    clean_txt = re.sub(r'@[A-Za-z0-9]+', '', clean_txt)
    # 6. Remove hashtags
    clean_txt = re.sub(r'#', '', clean_txt)
    # 7. Remove special characters and punctuation (use on clean_txt)
    clean_txt = re.sub(r'[^A-Za-z0-9\s]+', '', clean_txt)
    # 8. Remove extra spaces
    clean_txt = re.sub(r'\s+', ' ', clean_txt).strip() # .strip() removes leading/trailing space
    # 9 I found NaN in some resumes
    clean_txt = re.sub(r'NaN', ' ', clean_txt).strip() # .strip() removes leading/trailing space
    return clean_txt


# =========================== STREAMLIT UI ===========================
st.title("CV Parser")

uploaded_file = st.file_uploader("Upload Resume:", type=["txt", "pdf"])

if uploaded_file:
    resume_text = ""

    # ---- Handle TXT Files ----
    if uploaded_file.type == "text/plain":
        try:
            # For TXT, read content directly
            resume_text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            resume_text = uploaded_file.read().decode("latin-1")

        st.subheader("Uploaded Resume (Text)")
        st.text_area("File Content", resume_text, height=400)

    # ---- Handle PDF Files (FIXED LOGIC) ----
    elif uploaded_file.type == "application/pdf":
        # 1. Read the file into a bytes buffer, ensuring pointer is reset
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        # 2. Extract text using io.BytesIO for robustness with PyPDF2
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                # Add pages, using empty string if extraction fails
                resume_text += page.extract_text() or ""
        except Exception as e:
            # Display a user-friendly error if PyPDF2 fails
            st.error(f"Error reading PDF contents: {e}")

        # 3. Display PDF in app using the base64 encoded bytes
        st.subheader("Uploaded Resume (PDF Viewer)")
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

    # ---- Handle Empty or Invalid Extraction ----
    if not resume_text.strip():
        st.warning("No readable text found in the uploaded file.")
    else:
        st.success("File uploaded and processed successfully!")
        with st.expander("View Extracted Text"):
            # Display truncated text for brevity in the expander
            st.text_area("Extracted Text", resume_text[:3000], height=300)

        # ===================== MODEL 1 PREDICTION =====================
        resume_text_cleaned = clean_text(resume_text)
        vectorized_resume = tfidf.transform([resume_text_cleaned])
        prediction_ML_model1 = clf1.predict(vectorized_resume)[0]
        category_mapping = dict(enumerate(le.classes_))
        result_ML_model1 = category_mapping.get(prediction_ML_model1, "Unknown")
        st.success(f"Predicted Job Category by **ML_model1**: {result_ML_model1}")

        # ===================== MODEL 2 PREDICTION =====================
        prediction_ML_model2 = clf2.predict(vectorized_resume)[0]
        result_ML_model2 = category_mapping.get(prediction_ML_model2, "Unknown")
        st.success(f"Predicted Job Category by **ML_model2**: {result_ML_model2}")

        # ===================== MODEL 1 EVALUATION TOGGLE =====================
        if "show_eval_ml1" not in st.session_state:
            st.session_state["show_eval_ml1"] = False

        button_label_1 = (
            "Hide ML_model1 Evaluation" if st.session_state["show_eval_ml1"]
            else "Show ML_model1 Evaluation"
        )

        if st.button(button_label_1):
            st.session_state["show_eval_ml1"] = not st.session_state["show_eval_ml1"]

        if st.session_state["show_eval_ml1"]:
            y_pred1 = clf1.predict(x_test)
            y_proba1 = clf1.predict_proba(x_test)

            cm = confusion_matrix(y_test, y_pred1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
                ax=ax, cmap='Blues', xticks_rotation='vertical'
            )
            ax.set_title("Confusion Matrix (ML_model1)")
            fig.tight_layout()
            st.pyplot(fig)

            accuracy1 = accuracy_score(y_test, y_pred1)
            precision1 = precision_score(y_test, y_pred1, average='weighted', zero_division=0)
            recall1 = recall_score(y_test, y_pred1, average='weighted', zero_division=0)
            f1_score1 = f1_score(y_test, y_pred1, average='weighted')
            auc1 = roc_auc_score(y_test, y_proba1, multi_class='ovr')
            top3_acc1 = top_k_accuracy_score(y_test, y_proba1, k=3)

            st.subheader("Model 1 Evaluation Metrics (Test Set)")
            st.write(f"**Accuracy:** {accuracy1:.4f}")
            st.write(f"**Precision:** {precision1:.4f}")
            st.write(f"**Recall:** {recall1:.4f}")
            st.write(f"**F1 Score:** {f1_score1:.4f}")
            st.write(f"**AUC:** {auc1:.4f}")
            st.write(f"**Top-3 Accuracy:** {top3_acc1:.4f}")

        # ===================== MODEL 2 EVALUATION TOGGLE =====================
        if "show_eval_ml2" not in st.session_state:
            st.session_state["show_eval_ml2"] = False

        button_label_2 = (
            "Hide ML_model2 Evaluation" if st.session_state["show_eval_ml2"]
            else "Show ML_model2 Evaluation"
        )

        if st.button(button_label_2):
            st.session_state["show_eval_ml2"] = not st.session_state["show_eval_ml2"]

        if st.session_state["show_eval_ml2"]:
            y_pred2 = clf2.predict(x_test)
            y_proba2 = clf2.predict_proba(x_test)

            cm = confusion_matrix(y_test, y_pred2)
            fig, ax = plt.subplots(figsize=(10, 10))
            ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
                ax=ax, cmap='Blues', xticks_rotation='vertical'
            )
            ax.set_title("Confusion Matrix (ML_model2)")
            fig.tight_layout()
            st.pyplot(fig)

            accuracy2 = accuracy_score(y_test, y_pred2)
            precision2 = precision_score(y_test, y_pred2, average='weighted', zero_division=0)
            recall2 = recall_score(y_test, y_pred2, average='weighted', zero_division=0)
            f1_score2 = f1_score(y_test, y_pred2, average='weighted')
            auc2 = roc_auc_score(y_test, y_proba2, multi_class='ovr')
            top3_acc2 = top_k_accuracy_score(y_test, y_proba2, k=3)

            st.subheader("Model 2 Evaluation Metrics (Test Set)")
            st.write(f"**Accuracy:** {accuracy2:.4f}")
            st.write(f"**Precision:** {precision2:.4f}")
            st.write(f"**Recall:** {recall2:.4f}")
            st.write(f"**F1 Score:** {f1_score2:.4f}")
            st.write(f"**AUC:** {auc2:.4f}")
            st.write(f"**Top-3 Accuracy:** {top3_acc2:.4f}")
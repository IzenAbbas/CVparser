import streamlit as st
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import io
import base64
import json
from PyPDF2 import PdfReader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dropout=0.3):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, lstm_out):
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights

class BiLSTMClassifier(nn.Module):
    """BiLSTM + Attention"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=0,
                 lstm_layers=2, embed_dropout=0.2, fc_dropout1=0.5, fc_dropout2=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3,
            batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim, attn_dropout=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 192)
        self.dropout1 = nn.Dropout(fc_dropout1)
        self.bn2 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192, 96)
        self.dropout2 = nn.Dropout(fc_dropout2)
        self.fc3 = nn.Linear(96, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        context, attn_weights = self.attention(lstm_out)
        x = self.bn1(context)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class HybridBiLSTM_CNN_NoAttention(nn.Module):
    """BiLSTM + CNN (No Attention)"""
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        padding_idx=0,
        lstm_layers=2,
        cnn_filters=128,
        kernel_sizes=(2,3,4),
        embed_dropout=0.2,
        fc_dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=cnn_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        lstm_vec_dim = hidden_dim * 2 
        cnn_vec_dim = cnn_filters * len(kernel_sizes)
        fusion_dim = lstm_vec_dim + cnn_vec_dim
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        lstm_out, (h_n, _) = self.lstm(emb)
        lstm_vec = torch.cat((h_n[-2], h_n[-1]), dim=1)  
        cnn_input = lstm_out.permute(0, 2, 1) 
        cnn_feats = [
            torch.max(F.relu(conv(cnn_input)), dim=2)[0]
            for conv in self.convs
        ]
        cnn_vec = torch.cat(cnn_feats, dim=1) 
        fused = torch.cat([lstm_vec, cnn_vec], dim=1)
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionModel3(nn.Module):
    """Attention layer for Model 3 (no dropout, returns context only)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

class HybridBiLSTM_CNN(nn.Module):
    """BiLSTM + CNN + Attention (Hybrid Model)"""
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        padding_idx=0,
        lstm_layers=2,
        cnn_filters=128,
        kernel_sizes=(2,3,4),
        embed_dropout=0.2,
        fc_dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=lstm_layers, dropout=0.3,
            batch_first=True, bidirectional=True
        )
        self.attention = AttentionModel3(hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=cnn_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        lstm_vec_dim = hidden_dim * 2
        cnn_vec_dim = cnn_filters * len(kernel_sizes)
        fusion_dim = lstm_vec_dim + cnn_vec_dim
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)
        lstm_out, _ = self.lstm(emb) 
        lstm_vec = self.attention(lstm_out)
        cnn_input = lstm_out.permute(0, 2, 1) 
        cnn_feats = [
            torch.max(F.relu(conv(cnn_input)), dim=2)[0]
            for conv in self.convs
        ]
        cnn_vec = torch.cat(cnn_feats, dim=1) 
        fused = torch.cat([lstm_vec, cnn_vec], dim=1)
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_resources_v2():
    word2idx = pickle.load(open(f"{BASE_PATH}/word2idx.pkl", "rb"))
    idx2word = pickle.load(open(f"{BASE_PATH}/idx2word.pkl", "rb"))
    le = pickle.load(open(f"{BASE_PATH}/le.pkl", "rb"))
    tfidf = pickle.load(open(f"{BASE_PATH}/tfidf.pkl", "rb"))

    vocab_size = len(word2idx)
    num_classes = len(le.classes_)
    embed_dim = 128
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model1.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM_Attention.pt", map_location=device))
    model1 = model1.to(device)
    model1.eval()
    model2 = HybridBiLSTM_CNN_NoAttention(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model2.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM_CNN.pt", map_location=device))
    model2 = model2.to(device)
    model2.eval()
    
    model3 = HybridBiLSTM_CNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    model3.load_state_dict(torch.load(f"{BASE_PATH}/BiLSTM_CNN_Attention.pt", map_location=device))
    model3 = model3.to(device)
    model3.eval()
    clf1 = pickle.load(open(f"{BASE_PATH}/clf1.pkl", "rb"))
    clf2 = pickle.load(open(f"{BASE_PATH}/clf2.pkl", "rb"))
    clf3 = pickle.load(open(f"{BASE_PATH}/clf3_rf.pkl", "rb"))

    dl_history = {}
    try:
        dl_history["BiLSTM+Attention"] = json.load(open(f"{BASE_PATH}/history_model1.json", "r"))
        dl_history["BiLSTM+CNN"] = json.load(open(f"{BASE_PATH}/history_model2.json", "r"))
        dl_history["BiLSTM+CNN+Attention"] = json.load(open(f"{BASE_PATH}/history_model3.json", "r"))
    except FileNotFoundError:
        pass
    
    ml_history = {}
    transformer_history = {}
    try:
        transformer_history["Transformer (DistilBERT)"] = json.load(open(f"{BASE_PATH}/transformer_model/transformer_history.json", "r"))
    except FileNotFoundError:
        pass

    transformer_tokenizer = None
    transformer_model = None
    transformer_path = os.path.join(BASE_PATH, "transformer_model")
    try:
        transformer_tokenizer = DistilBertTokenizerFast.from_pretrained(transformer_path)
        transformer_model = DistilBertForSequenceClassification.from_pretrained(transformer_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        transformer_model.eval()
    except Exception as e:
        transformer_tokenizer = None
        transformer_model = None

    return word2idx, idx2word, le, tfidf, model1, model2, model3, clf1, clf2, clf3, dl_history, ml_history, transformer_history, transformer_tokenizer, transformer_model

word2idx, idx2word, le, tfidf, model1, model2, model3, clf1, clf2, clf3, dl_history, ml_history, transformer_history, tokenizer_trans, transformer_model = load_resources_v2()

MAX_LEN = 500
EMBED_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_resume(text):
    if not text:
        return ""
    s = str(text)
    s = re.sub(r'\S+@\S+', ' ', s)
    s = re.sub(r'http\S+', ' ', s)
    replacements = {
        "C++": "CPLUSPLUS", "c++": "CPLUSPLUS",
        "C#": "CSHARP", "c#": "CSHARP",
        ".NET": "DOTNET", ".net": "DOTNET",
        "Node.js": "NODEJS", "node.js": "NODEJS"
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r'[^A-Za-z0-9+\#\./\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    inv = {v: k.lower() for k, v in replacements.items()}
    for k, v in inv.items():
        s = s.replace(k, v)
    return s.lower().strip()


def text_to_seq(text):
    seq = [word2idx.get(w, 1) for w in text.split()]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return torch.tensor([seq], dtype=torch.long).to(device)


def transformer_predict(text, tokenizer, model):
    if tokenizer is None or model is None:
        return None
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return probs

vocab_size_actual = len(word2idx)
num_classes = len(le.classes_)

st.title("Resume Job Category Prediction")

ml_models = {
    "Logistic Regression": clf1,
    "Linear SVM": clf2,
    "Random Forest": clf3
}
dl_models = {
    "BiLSTM+Attention": model1,
    "BiLSTM+CNN": model2,
    "BiLSTM+CNN+Attention": model3
}

page = st.sidebar.selectbox("Choose a Page", ["Prediction", "Model Evaluation"]) 

if page == "Prediction":
    st.header("Upload Resume for Prediction")
    uploaded_file = st.file_uploader("Upload Resume:", type=["txt", "pdf"])
    
    resume_text = ""
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            try:
                resume_text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                resume_text = uploaded_file.read().decode("latin-1")
            st.subheader("Uploaded Resume (Text)")
            st.text_area("File Content", resume_text, height=400)
        
        elif uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            try:
                pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in pdf_reader.pages:
                    resume_text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Failed to read PDF file: {e}")

            st.subheader("Uploaded Resume (PDF Viewer)")
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

            st.subheader("Extracted Text")
            st.text_area("Content", resume_text, height=300)
    if not resume_text:
        st.info("Using sample resume for demonstration.")
        resume_text = """Skills * Programming Languages: Python (pandas, numpy, scipy, scikit-learn, matplotlib), R, Sql, Spark, Scala. 
        Machine learning: Deep Learning, CNN, RNN, Transformers, Regression, SVM, Random Forest, Ensemble Methods, NLP, Time Series.
        Databases: MySQL, MongoDB, PowerBI, AWS, GCP.
        Education: MS Data Science, Stanford University.
        Experience: Senior Data Scientist at Tech Innovations."""
        st.text_area("Sample Resume", resume_text, height=200)

    if resume_text:
        cleaned_text = clean_resume(resume_text)
        
        st.subheader("Cleaned Text")
        st.text_area("Cleaned Content", cleaned_text, height=200)
  
        seq = text_to_seq(cleaned_text)
        tfidf_vec = tfidf.transform([cleaned_text])
        
        st.subheader("Predictions")
     
        st.markdown("### Deep Learning Models")
        cols = st.columns(len(dl_models))
        for i, (name, model) in enumerate(dl_models.items()):
            with cols[i]:
                with torch.no_grad():
                    logits = model(seq)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                top1_idx = np.argmax(probs)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.info(f"**{name}**\n\n{top1_label}\n({probs[top1_idx]:.4f})")
                
                with st.expander(f"Top-5 ({name})"):
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")

        st.markdown("### Machine Learning Models")
        cols_ml = st.columns(len(ml_models))
        for i, (name, model) in enumerate(ml_models.items()):
            with cols_ml[i]:
                probs = model.predict_proba(tfidf_vec)[0]
                top1_idx = np.argmax(probs)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.success(f"**{name}**\n\n{top1_label}\n({probs[top1_idx]:.4f})")
                
                with st.expander(f"Top-5 ({name})"):
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")

        if tokenizer_trans is not None and transformer_model is not None:
            st.markdown("### Transformer Model")
            probs_t = transformer_predict(cleaned_text, tokenizer_trans, transformer_model)
            if probs_t is not None:
                top1_idx = np.argmax(probs_t)
                top1_label = le.inverse_transform([top1_idx])[0]
                st.warning(f"**DistilBERT Transformer**\n\n{top1_label}\n({probs_t[top1_idx]:.4f})")
                with st.expander("Top-5 (Transformer)"):
                    top5_idx = probs_t.argsort()[-5:][::-1]
                    top5_labels = le.inverse_transform(top5_idx)
                    top5_probs = probs_t[top5_idx]
                    for l, p in zip(top5_labels, top5_probs):
                        st.write(f"{l}: {p:.4f}")
        else:
            st.info("Transformer model not available in the resources folder.")

elif page == "Model Evaluation":
    st.header("Model Evaluation on Test Set")
    
    eval_cache_path = f"{BASE_PATH}/evaluation_results.pkl"
    results = None
    
    if os.path.exists(eval_cache_path):
        with st.spinner("Loading evaluation results..."):
            with open(eval_cache_path, 'rb') as f:
                results = pickle.load(f)
        st.info("✓ Evaluation results loaded from cache.")
    else:
        st.error("Evaluation results not found. Please run evaluate.py first to generate the results.")
  
    if results is not None:
        model_results = {k: v for k, v in results.items() if k != 'learning_curves'}

        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            name: {k: v for k, v in res.items() if k != "Confusion Matrix"}
            for name, res in model_results.items()
        })
        st.dataframe(metrics_df.style.highlight_max(axis=1))
   
        st.subheader("Confusion Matrices")
        tabs = st.tabs(list(model_results.keys()))
        for i, (name, res) in enumerate(model_results.items()):
            with tabs[i]:
                st.write(f"**{name}**")
                cm = res["Confusion Matrix"]
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                st.pyplot(fig)
     
        if dl_history:
            st.subheader("Training History - Deep Learning Models")
            for name, h in dl_history.items():
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_loss'], label='Train Loss')
                    ax.plot(h['val_loss'], label='Val Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{name} - Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_acc'], label='Train Acc')
                    ax.plot(h['val_acc'], label='Val Acc')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'{name} - Accuracy')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
       
        if transformer_history:
            st.subheader("Training History - Transformer Model")
            for name, h in transformer_history.items():
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_loss'], label='Train Loss')
                    ax.plot(h['val_loss'], label='Val Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{name} - Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(h['train_acc'], label='Train Acc')
                    ax.plot(h['val_acc'], label='Val Acc')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title(f'{name} - Accuracy')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        if not dl_history and not ml_history and not transformer_history:
            st.info("Note: Training and validation loss graphs are not available. Run the training scripts to generate history files.")

        if 'learning_curves' in results:
            st.subheader("Learning Curves - Machine Learning Models")
            for name, data in results['learning_curves'].items():
                st.markdown(f"**{name}**")
                
                train_sizes = data['train_sizes']
                train_scores = data['train_scores']
                test_scores = data['test_scores']
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Learning Curve ({name})")
                ax.set_ylim(0.7, 1.01)
                ax.set_xlabel("Training examples")
                ax.set_ylabel("Score")
                ax.grid()
                
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                
                ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
                ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
                ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
                ax.legend(loc="best")
                
                st.pyplot(fig)
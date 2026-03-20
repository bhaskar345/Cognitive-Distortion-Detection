import streamlit as st
import onnxruntime as ort
import numpy as np
import json
from tokenizers import Tokenizer

# -----------------------------
# Load everything (cached)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = Tokenizer.from_file("model/tokenizer.json")

    with open("model/config.json") as f:
        config = json.load(f)

    label_classes = np.load("model/label_classes.npy", allow_pickle=True)
    emotion_labels = np.load("model/emotion_labels.npy", allow_pickle=True)

    session = ort.InferenceSession("model/multitask_model.onnx")

    return tokenizer, config, label_classes, emotion_labels, session


tokenizer, config, label_classes, emotion_labels, session = load_model()
MAX_LEN = config["max_len"]

id2label = {i: l for i, l in enumerate(label_classes)}

emotion_threshold = 0.2

# -----------------------------
# Prediction function
# -----------------------------
def predict_onnx(text):
    encoded = tokenizer.encode(text)

    input_ids = encoded.ids[:MAX_LEN]
    attention_mask = encoded.attention_mask[:MAX_LEN]

    # pad manually
    pad_len = MAX_LEN - len(input_ids)

    input_ids = input_ids + [0] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    input_ids = np.array([input_ids], dtype=np.int64)
    attention_mask = np.array([attention_mask], dtype=np.int64)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    outputs = session.run(None, inputs)
    emotion_logits, distortion_logits, start_logits, end_logits = outputs

    # --- Distortion (softmax) ---
    exp_logits = np.exp(distortion_logits - np.max(distortion_logits, axis=1, keepdims=True))
    distortion_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    d_idx = int(np.argmax(distortion_probs[0]))
    distortion = id2label[d_idx]
    confidence = round(float(distortion_probs[0][d_idx]) * 100,2)

    # --- Span ---
    offsets = encoded.offsets
    s = int(np.argmax(start_logits[0]))
    e = int(np.argmax(end_logits[0]))

    if s > e:
        s, e = e, s

    span = text[offsets[s][0]:offsets[e][1]] if offsets[s][0] < len(text) else ""

    # --- Emotion (multi-label sigmoid) ---
    emotion_probs = 1 / (1 + np.exp(-emotion_logits[0]))

    emotions = [
        {
            "label": emotion_labels[i],
            "score": round(float(emotion_probs[i]) * 100, 2)
        }
        for i in range(len(emotion_probs))
        if emotion_probs[i] > emotion_threshold
    ]

    return distortion, confidence, span, emotions


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Cognitive Distortion Detector", layout="centered")

st.title("🧠 Cognitive Distortion Detector")
st.write("Enter a patient statement to detect distortion, emotion, and highlighted span.")

text = st.text_area("Input Text", height=120,
                     placeholder="e.g. They are ignoring me because they don’t like me anymore.")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        distortion, confidence, span, emotions = predict_onnx(text)

        st.subheader("📌 Results")

        st.markdown(f"**Distortion:** `{distortion}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        st.markdown("**Extracted Distorted Span:**")
        st.info(span if span else "No valid span detected")

        st.markdown("**Detected Emotions:**")
        if emotions:
            for e in emotions:
                st.write(f"- {e['label']} : {e['score']:.2f}%")
        else:
            st.write("No strong emotions detected")
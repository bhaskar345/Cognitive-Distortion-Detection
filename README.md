# 🧠 Cognitive Distortion Detector

> **Cognitive distortions** are irrational or exaggerated thought patterns that negatively influence emotions and behavior — common in anxiety, depression, and stress. They subtly warp our perception of reality, making situations seem worse than they actually are. This app uses AI to identify and explain these thought patterns in real time, helping users and clinicians gain deeper self-awareness.

---

## ✨ What This App Does

The **Cognitive Distortion Detector** is a lightweight, AI-powered web application built with **Streamlit** and powered by a custom **multitask ONNX model**. Given a patient or user statement, it simultaneously:

| Output | Description |
|---|---|
| 🏷️ **Distortion Type** | Classifies the cognitive distortion (e.g., *Catastrophizing*, *Mind Reading*, *All-or-Nothing Thinking*) |
| 📊 **Confidence Score** | Shows how confident the model is in its prediction (%) |
| 🔍 **Distorted Span** | Highlights the exact phrase in the input that triggered the distortion |
| 💬 **Detected Emotions** | Lists emotions present in the text (e.g., fear, sadness, anger) with scores |

The model supports **12 distortion classes** and **28 emotion labels**, running entirely offline via ONNX — no API keys, no internet required after setup.

[Download Model](https://drive.google.com/drive/folders/14m1CnT6-LUVMlFHHde93PMyc6ofIXw_7?usp=sharing)

---

## 🗂️ Project Structure

```
Cognitive-Distortion-Detection/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── dataset.json            # Labeled training dataset
├── training.ipynb          # Model training notebook (Google Colab)
│
└── model/
    ├── multitask_model.onnx        # ONNX model weights
    ├── multitask_model.onnx.data   # External model data
    ├── tokenizer.json              # HuggingFace tokenizer
    ├── tokenizer_config.json       # Tokenizer configuration
    ├── config.json                 # Model config (max_len, num_labels)
    ├── label_classes.npy           # Distortion class labels
    └── emotion_labels.npy          # Emotion class labels
```

---

## ⚙️ Installation

### Prerequisites

- Python **3.8+**
- `pip` package manager

### Step 1 — Clone or Download the Project

```bash
git clone https://github.com/bhaskar345/Cognitive-Distortion-Detection.git
cd Cognitive-Distortion-Detection
```

Or simply download and extract the ZIP, then open a terminal in the project folder.

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv env

# Activate it
# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `onnxruntime` | Run the ONNX model |
| `tokenizers` | Fast HuggingFace tokenizer |
| `numpy` | Numerical operations |

---

## 🚀 Running the App

Make sure your virtual environment is activated, then run:

```bash
streamlit run app.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

> 💡 **Note:** Ensure the `model/` folder is present in the same directory as `app.py`. The app loads all model files from this folder at startup.

---

## 🖥️ Usage

1. **Open the app** in your browser after running the command above.
2. **Type or paste** a statement into the text box.
   - Example: *"They are ignoring me because they don't like me anymore."*
3. **Click "Analyze"** to run the model.
4. **View the results:**
   - The detected **distortion type** and its confidence score
   - The **highlighted span** — the exact phrase causing the distortion
   - Any **emotions** detected above the threshold (20% score)

---

## 🏋️ Training

The model was trained on **Google Colab (T4 GPU)** using `training.ipynb`.

### Architecture

A **multitask transformer model** fine-tuned on top of a pretrained BERT-based backbone. Four task heads are attached to the shared encoder:

```
Input Text
    │
    ▼
[Pretrained BERT Encoder]
    │
    ├──► Distortion Head   →  Softmax over 13 classes
    ├──► Emotion Head      →  Sigmoid over 28 emotion labels (multi-label)
    ├──► Span Start Head   →  Token-level classification (QA-style)
    └──► Span End Head     →  Token-level classification (QA-style)
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv activated |
| Model files not found | Ensure the `model/` folder is in the same directory as `app.py` |
| App doesn't open in browser | Manually visit `http://localhost:8501` |
| Slow first load | Normal — Streamlit caches the model on first run (`@st.cache_resource`) |
| ONNX runtime error | Ensure `onnxruntime` is installed, not `onnxruntime-gpu` (unless on GPU) |

---
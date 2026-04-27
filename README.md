# 🧸 MySLM – Kids Story Chatbot

## 📌 Overview

MySLM is an **end-to-end AI chatbot application** that generates **simple, engaging, and child-friendly stories** using a custom-trained small language model (SLM).

The project demonstrates the full pipeline of:

* model development
* deployment
* interactive UI

---

## 🚀 Live Demo

* 🌐 **Streamlit App:**
  https://bbc858wvjbbvugtxgslfpr.streamlit.app/

---

## 🤖 Model Hosting

* 🤗 **Hugging Face Model:**
  https://huggingface.co/Sachin0803/myslm

---

## 💻 Source Code

* 🔗 **GitHub Repository:**
  https://github.com/Sachindtu402/my_slm

---

## 🧠 Features

* 📖 Generates simple and fun stories for children
* 💬 Interactive chat-based interface
* ⚙️ Adjustable generation parameters (temperature, max tokens)
* 🎯 Optimized for easy-to-understand language
* ⚡ Fast inference with cached model loading

---

## 🏗️ Tech Stack

* **Python**
* **PyTorch** (Model training & inference)
* **Streamlit** (Frontend UI)
* **Hugging Face Hub** (Model hosting)
* **tiktoken** (Tokenization)

---

## 📂 Project Structure

```
my_slm/
 ├── app.py              # Streamlit application
 ├── model.py            # Custom GPT model architecture
 ├── requirements.txt    # Dependencies
 └── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sachindtu402/my_slm
cd my_slm
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Add Hugging Face Token (for Streamlit)

Create `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "your_huggingface_token"
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🎯 Usage

* Open the app
* Type a prompt like:

  > "Tell me a story about a cat"
* Get a simple, fun story generated instantly

---

## 🧠 Model Details

* Custom GPT-style architecture
* Vocabulary size: 50,257
* Context length: 128
* Lightweight Small Language Model (SLM)
* Trained on story-style dataset (TinyStories-inspired)

---

## 🔒 Security

* Hugging Face token stored securely using **Streamlit Secrets**
* No hardcoded credentials

---

## 🚀 Future Improvements

* Improve story coherence
* Add voice narration (text-to-speech)
* Enhance UI (storybook style)
* Add memory and personalization

---

## 🙌 Acknowledgements

* Hugging Face
* Streamlit
* PyTorch



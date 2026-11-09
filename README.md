# Scriptify

*From Handwriting to Smart Learning.*

Scriptify is a Python-based application that converts handwritten notes into editable digital text and helps students study smarter with AI-powered learning tools.

## Features
- Handwritten text recognition using Microsoft TrOCR.
- Image preprocessing and line segmentation using OpenCV.
- Editable extracted text with TXT and PDF export.
- AI-generated summaries, important information, and key points.
- Interactive MCQ quizzes with automatic scoring.
- Topic-wise revision and question answering.

## Tech Stack
Python, Streamlit, OpenCV, TrOCR, PyTorch, Groq API, NumPy.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Configuration:** Add your Groq API key to `.streamlit/secrets.toml` before using the Study Assistant.

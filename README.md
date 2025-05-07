# Multilingual Speech-to-Text App (Using Deep Learning & NLP)

This project is a **multilingual speech recognition system** that converts spoken language (from WAV audio files) into text using a custom-trained deep learning model. It supports **multiple languages** including English and Hindi.

## Features

* Record speech directly from the browser (HTML + JavaScript)
* Accept `.wav` audio files for processing
* DeepSpeech2-like architecture with CTC loss (trained in TensorFlow/Keras)
* Multilingual transcription support
* Django-based backend for handling file upload and prediction
* Real-time transcription output in the frontend

---

## Tech Stack

| Component       | Technology                    |
| --------------- | ----------------------------- |
| Frontend        | HTML, CSS, JavaScript         |
| Audio Recording | MediaRecorder.js              |
| Backend API     | Django, Django REST Framework |
| Model Framework | TensorFlow/Keras              |
| Data Processing | NumPy, Pandas                 |
| Speech Model    | Bi-GRU with CTC Loss          |


# 🖼️ Image Captioning with Deep Learning

This project implements and evaluates an image captioning model built with **PyTorch**. It uses a classic **CNN-LSTM architecture** to generate descriptive captions for images. The project includes scripts for training, inference, and a user-friendly web interface built with **Gradio**.

---

## 🚀 Features

- **CNN-LSTM Architecture**  
  Utilizes a pre-trained **ResNet-50** as the image feature encoder and an **LSTM** network as the decoder.

- **Validation & Early Stopping**  
  Includes validation loop and early stopping to monitor performance and avoid overfitting.

- **Beam Search Decoding**  
  Generates coherent captions using beam search instead of greedy decoding.

- **Gradio Web UI**  
  A simple, interactive web application to test the model by uploading an image.

- **Optimized for Apple Silicon (M1/M2/M3)**  
  Configured to leverage **MPS** for GPU acceleration on modern MacBooks.

---

## 📁 Project Structure

```

image-captioning-project/
├── app/
│   └── app.py                      # Gradio web application
├── data/
│   └── P5 Image Captioning/       # Holds the Flickr8k dataset
├── models/
│   └── best\_model\_cnn\_lstm.pth.tar # Best trained model checkpoint
├── src/
│   ├── cnn\_lstm\_model.py          # Model architecture (Encoder + Decoder)
│   ├── data\_loader.py             # PyTorch Dataset and DataLoader
│   ├── inference.py               # Inference script
│   └── train.py                   # Model training script
├── .gitignore
└── README.md

````

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd image-captioning-project
````

2. **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download Data:**

* Download the Flickr8k dataset from the provided Dropbox link.
* Unzip and place the folders `Flicker8k_Dataset` and `Flickr8k_text` inside:

```
data/P5 Image Captioning/
```

---

## 🧪 How to Run

### 🏋️‍♀️ Training

Train the model from scratch:

```bash
python src/train.py
```

The best-performing model checkpoint will be saved in the `models/` directory.

### 🧾 Inference (Command Line)

Run caption generation on a test image:

```bash
python src/inference.py
```

This will display the image and the generated caption.

### 🌐 Launching the Web App

Start the Gradio interface:

```bash
python app/app.py
```

Then open your browser and go to:

```
http://127.0.0.1:7860
```

---

## 🔮 Future Improvements

* **Vision Transformer (ViT) Encoder**
  Replace ResNet with a pre-trained ViT to capture global image features.

* **Hugging Face Transformer Decoder**
  Use a pre-trained decoder like **GPT-2** or **BART** for more fluent text generation.

* **Attention Mechanism**
  Integrate Bahdanau or Luong attention in the LSTM decoder for better alignment.

* **Advanced Evaluation Metrics**
  Use metrics like **BLEU**, **ROUGE**, and **CIDEr** for richer model evaluation.

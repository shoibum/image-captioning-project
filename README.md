Image Captioning with Deep Learning
This project implements and evaluates an image captioning model built with PyTorch. It uses a classic CNN-LSTM architecture to generate descriptive captions for given images. The project includes scripts for training, inference, and a user-friendly web interface built with Gradio.

Features
CNN-LSTM Architecture: Utilizes a pre-trained ResNet-50 as the image feature encoder and an LSTM network as the caption-generating decoder.

Validation & Early Stopping: The training script includes a validation loop to monitor performance and early stopping to prevent overfitting and save the best model.

Beam Search Decoding: Implements beam search for inference to generate higher-quality, more coherent captions compared to greedy search.

Gradio Web UI: A simple and interactive web application to easily test the model by uploading an image.

Optimized for Apple Silicon: The project is configured to leverage MPS for GPU acceleration on modern MacBooks.

Project Structure
image-captioning-project/
├── app/
│   └── app.py              # Gradio web application
├── data/
│   └── P5 Image Captioning/  # Holds the Flickr8k dataset
├── models/
│   └── best_model_cnn_lstm.pth.tar # Best trained model checkpoint
├── src/
│   ├── cnn_lstm_model.py     # Model architecture (Encoder + Decoder)
│   ├── data_loader.py        # PyTorch Dataset and DataLoader
│   ├── inference.py          # Script to run inference on a random test image
│   └── train.py              # Script to train the model
├── .gitignore
└── README.md

Setup & Installation
1. Clone the repository:

git clone <your-repository-url>
cd image-captioning-project

2. Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Download Data:
Download the Flickr8k dataset from the provided link and unzip it. Place the Flicker8k_Dataset and Flickr8k_text folders inside data/P5 Image Captioning/.

How to Run
1. Training
To train the model from scratch, run the following command. The script will save the best-performing model checkpoint in the models/ directory.

python src/train.py

2. Inference (Command Line)
To test the trained model on a random image from the test set, run:

python src/inference.py

This will display the image and its generated caption.

3. Launching the Web App
To start the interactive Gradio UI, run:

python app/app.py

Then, open your web browser to http://127.0.0.1:7860.

Future Improvements
Vision Transformer (ViT) Encoder: Replace the ResNet CNN encoder with a pre-trained Vision Transformer. ViT can often capture more global context from an image, potentially leading to more descriptive captions.

Hugging Face Transformer Decoder: Replace the custom LSTM decoder with a powerful pre-trained language model from Hugging Face, such as GPT-2 or BART. This would allow us to leverage large-scale language understanding for more fluent and accurate text generation.

Attention Mechanism: Enhance the current LSTM decoder by adding a Bahdanau or Luong attention mechanism. This would allow the decoder to "focus" on different parts of the image as it generates each word, which is known to significantly improve performance.

Advanced Evaluation Metrics: Implement more sophisticated evaluation metrics beyond loss, such as BLEU, ROUGE, and CIDEr, to better compare the performance of different models against human-generated captions.

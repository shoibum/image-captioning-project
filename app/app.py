import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cnn_lstm_model import CNNtoRNN
from data_loader import Vocabulary

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = None
VOCAB = None

def load_model():
    global MODEL, VOCAB
    if MODEL is not None:
        return

    checkpoint_path = os.path.join('models', 'best_model_cnn_lstm.pth.tar')
    
    print(f"Loading model from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure you have trained the model and the file exists.")
        return

    VOCAB = checkpoint["vocab"]
    vocab_size = len(VOCAB)
    
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    
    MODEL = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(DEVICE)
    MODEL.load_state_dict(checkpoint["state_dict"])
    MODEL.eval()
    print("Model loaded successfully.")

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_caption(image_pil):
    if MODEL is None or VOCAB is None:
        return "Model is not loaded. Please check the console for errors."

    transform = get_image_transform()
    image_tensor = transform(image_pil).to(DEVICE)
    
    generated_caption = MODEL.caption_image(image_tensor, VOCAB, device=DEVICE, beam_width=5)
    
    # More robustly filter out special tokens
    words = generated_caption.split(' ')
    clean_words = [word for word in words if word.lower() not in ['<sos>', '<eos>', '<unk>']]
    cleaned_caption = ' '.join(clean_words).strip()
    
    if cleaned_caption:
        cleaned_caption = cleaned_caption[0].upper() + cleaned_caption[1:]
    else:
        cleaned_caption = "Could not generate a caption."
    
    return cleaned_caption

if __name__ == "__main__":
    load_model()
    
    iface = gr.Interface(
        fn=predict_caption,
        inputs=gr.Image(type="pil", label="Upload an Image"),
        outputs=gr.Textbox(label="Generated Caption"),
        title="Image Captioning with CNN-LSTM",
        description="Upload any image and the model will generate a caption for it. This model was trained on the Flickr8k dataset using a ResNet-50 Encoder and LSTM Decoder with Beam Search.",
        allow_flagging="never"
    )

    iface.launch()
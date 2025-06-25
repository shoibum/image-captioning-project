# src/inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import random

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from cnn_lstm_model import CNNtoRNN
from data_loader import Vocabulary

def load_checkpoint(filepath, device):
    print(f"=> Loading checkpoint from {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        return checkpoint
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {filepath}")
        print("Ensure you are running this script from the project's root directory.")
        return None

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join('models', 'best_model_cnn_lstm.pth.tar')
    
    text_files_dir = os.path.join('data', 'P5 Image Captioning', 'Flickr8k_text')
    test_split_file = os.path.join(text_files_dir, 'Flickr_8k.testImages.txt')
    
    with open(test_split_file, 'r') as f:
        test_images = [line.strip() for line in f]
    
    image_to_test = random.choice(test_images)
    print(f"Testing with random image: {image_to_test}")
    
    base_data_path = os.path.join('data', 'P5 Image Captioning')
    image_dir = os.path.join(base_data_path, 'Flicker8k_Dataset')
    image_path = os.path.join(image_dir, image_to_test)
    

    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint is None:
        return
        
    vocab = checkpoint["vocab"]
    vocab_size = len(vocab)
    
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    transform = get_image_transform()
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image)
    
    generated_caption = model.caption_image(transformed_image, vocab, device=device, beam_width=5)

    print("\n" + "="*50)
    print("Generated Caption:")

    cleaned_caption = ' '.join(generated_caption.split(' ')[1:-1])
    print(cleaned_caption)
    print("="*50)
    
    plt.imshow(image)
    plt.title(f"Caption: {cleaned_caption}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
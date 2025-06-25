# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np

from data_loader import get_loader
from cnn_lstm_model import CNNtoRNN

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves the model checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def main():
    """Main training and validation loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    batch_size = 64
    patience = 5 
    
    base_data_path = os.path.join('data', 'P5 Image Captioning')
    image_dir = os.path.join(base_data_path, 'Flicker8k_Dataset')
    text_files_dir = os.path.join(base_data_path, 'Flickr8k_text')
    captions_file = os.path.join(text_files_dir, 'Flickr8k.token.txt')
    train_split_file = os.path.join(text_files_dir, 'Flickr_8k.trainImages.txt')
    val_split_file = os.path.join(text_files_dir, 'Flickr_8k.devImages.txt')


    train_loader, train_dataset = get_loader(
        image_dir, captions_file, train_split_file, transform, batch_size,
        num_workers=0, shuffle=True, pin_memory=False, mode='train'
    )
    val_loader, val_dataset = get_loader(
        image_dir, captions_file, val_split_file, transform, batch_size,
        num_workers=0, shuffle=False, pin_memory=False, mode='val', vocab=train_dataset.vocab
    )

    vocab_size = len(train_dataset.vocab)
    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader), leave=False)
        train_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}] Training")

        for imgs, captions in train_loop:
            imgs = imgs.to(device)
            captions = captions.permute(1, 0).to(device)

            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

        model.eval()
        val_losses = []
        val_loop = tqdm(val_loader, total=len(val_loader), leave=False)
        val_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}] Validation")

        with torch.no_grad():
            for imgs, captions in val_loop:
                imgs = imgs.to(device)
                captions = captions.permute(1, 0).to(device)
                
                outputs = model(imgs, captions)
                val_loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": train_dataset.vocab,
            }
            if not os.path.exists('models'):
                os.makedirs('models')
            save_checkpoint(checkpoint, filename=os.path.join('models', 'best_model_cnn_lstm.pth.tar'))
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered. Halting training.")
            break

if __name__ == "__main__":
    main()

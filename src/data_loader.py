# src/data_loader.py
import os
import pandas as pd
import spacy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from collections import Counter
import torchvision.transforms as transforms

try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' for spacy...")
    from spacy.cli import download
    download("en_core_web_sm")
    spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, split_file, transform=None, freq_threshold=5, mode='train', vocab=None):

        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        all_captions_df = pd.read_csv(captions_file, sep='\t', names=['image', 'caption'])
        with open(split_file, 'r') as f:
            split_image_names = {line.strip() for line in f}

        all_captions_df['image_name'] = all_captions_df['image'].apply(lambda x: x.split('#')[0])
        
        self.df = all_captions_df[all_captions_df['image_name'].isin(split_image_names)].reset_index(drop=True)

        if self.mode == 'train':
            self.captions = self.df["caption"]
            if vocab is None:
                self.vocab = Vocabulary(freq_threshold)
                self.vocab.build_vocabulary(self.captions.tolist())
            else:
                self.vocab = vocab
        else:
            assert vocab is not None, "A vocabulary must be provided for validation/test sets."
            self.vocab = vocab
        
        self.image_names = self.df['image_name'].unique().tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        captions_for_image = self.df[self.df['image_name'] == image_name]['caption'].tolist()

        caption = captions_for_image[torch.randint(0, len(captions_for_image), (1,)).item()]
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption), captions_for_image

class Collate:
    """Pads captions in a batch to the same length."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(image_dir, captions_file, split_file, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True, mode='train', vocab=None):
    dataset = FlickrDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        split_file=split_file,
        transform=transform,
        mode=mode,
        vocab=vocab
    )
    
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if mode=='train' else False, 
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    BASE_DATA_PATH = os.path.join('data', 'P5 Image Captioning')
    IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'Flicker8k_Dataset') 
    TEXT_FILES_DIR = os.path.join(BASE_DATA_PATH, 'Flickr8k_text')

    CAPTIONS_FILE = os.path.join(TEXT_FILES_DIR, 'Flickr8k.token.txt')
    TRAIN_SPLIT_FILE = os.path.join(TEXT_FILES_DIR, 'Flickr_8k.trainImages.txt')
    VAL_SPLIT_FILE = os.path.join(TEXT_FILES_DIR, 'Flickr_8k.devImages.txt')
    
    print("--- Loading Training Data ---")
    train_loader, train_dataset = get_loader(
        IMAGE_DIR, CAPTIONS_FILE, TRAIN_SPLIT_FILE, transform, batch_size=4, mode='train'
    )
    
    vocab = train_dataset.vocab
    print(f"Vocabulary Size: {len(vocab)}")

    print("\n--- Loading Validation Data ---")
    val_loader, val_dataset = get_loader(
        IMAGE_DIR, CAPTIONS_FILE, VAL_SPLIT_FILE, transform, batch_size=4, mode='val', vocab=vocab
    )
    
    print("\n--- Testing DataLoaders ---")
    train_images, train_captions = next(iter(train_loader))
    print(f"Train images batch shape: {train_images.shape}")
    print(f"Train captions batch shape: {train_captions.shape}")

    val_images, val_captions = next(iter(val_loader))
    print(f"Validation images batch shape: {val_images.shape}")
    print(f"Validation captions batch shape: {val_captions.shape}")
    print("\nData loader setup is correct.")

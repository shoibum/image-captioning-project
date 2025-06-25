# src/cnn_lstm_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(train_cnn)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        captions = captions[:, :-1] 
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
        
    def sample_beam_search(self, features, vocab, device, beam_width=5, max_len=20):
        start_idx = vocab.stoi["<SOS>"]
        end_idx = vocab.stoi["<EOS>"]
        
        inputs = features.unsqueeze(1)
        hidden_state, cell_state = None, None
        hiddens, (hidden_state, cell_state) = self.lstm(inputs, None)
        

        hidden_state = hidden_state.expand(-1, beam_width, -1)
        cell_state = cell_state.expand(-1, beam_width, -1)
        
        outputs = self.linear(hiddens.squeeze(1))
        log_probs = torch.log_softmax(outputs, dim=1)
        top_scores, top_words = log_probs[0].topk(beam_width, dim=0)

        sequences = [[[start_idx, w.item()] for w in top_words]]
        scores = top_scores.unsqueeze(1)
        
        completed_sequences = []
        completed_scores = []

        for _ in range(max_len):
            last_words = torch.tensor([seq[-1] for seq in sequences[0]], dtype=torch.long).to(device)
            inputs = self.embedding(last_words).unsqueeze(1)

            hiddens, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state))
            outputs = self.linear(hiddens.squeeze(1))
            log_probs = torch.log_softmax(outputs, dim=1)
            
            all_scores = scores + log_probs
            
            all_scores_flat = all_scores.view(-1)
            top_k_scores, top_k_indices = all_scores_flat.topk(beam_width, dim=0)
            
            beam_indices = torch.div(top_k_indices, len(vocab), rounding_mode='floor')
            word_indices = top_k_indices % len(vocab)
            
            new_sequences = []
            new_scores = []
            next_hidden_state, next_cell_state = [], []

            for i, (word_idx, beam_idx) in enumerate(zip(word_indices, beam_indices)):
                new_seq = sequences[0][beam_idx] + [word_idx.item()]
                
                if word_idx.item() == end_idx:
                    completed_sequences.append(new_seq)
                    completed_scores.append(top_k_scores[i].item())
                    beam_width -= 1
                else:
                    new_sequences.append(new_seq)
                    new_scores.append(top_k_scores[i])
                    next_hidden_state.append(hidden_state[:, beam_idx, :])
                    next_cell_state.append(cell_state[:, beam_idx, :])

            if beam_width == 0: break 
            
            sequences = [new_sequences]
            scores = torch.stack(new_scores).unsqueeze(1)
            hidden_state = torch.stack(next_hidden_state, dim=1)
            cell_state = torch.stack(next_cell_state, dim=1)
        
        if len(completed_sequences) > 0:
            normalized_scores = [s / len(seq) for s, seq in zip(completed_scores, completed_sequences)]
            best_seq_idx = np.argmax(normalized_scores)
            return completed_sequences[best_seq_idx]
        else:
            return sequences[0][0] if sequences and sequences[0] else [start_idx, end_idx]


class CNNtoRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, device, max_len=20, beam_width=5):

        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(device))
            caps_ids = self.decoder.sample_beam_search(features, vocabulary, device, beam_width, max_len)
        
        caption_words = [vocabulary.itos[idx] for idx in caps_ids]
        caption = ' '.join(caption_words)
        return caption

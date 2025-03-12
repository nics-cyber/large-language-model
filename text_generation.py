import os
import requests
import pylzma
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


DATA_URL = 'https://skylion007.github.io/OpenWebTextCorpus/openwebtext.tar.xz'
DATA_DIR = 'data'
EXTRACTED_FILE = os.path.join(DATA_DIR, 'openwebtext.txt')
TOKENIZER_FILE = 'bpe_tokenizer.json'
MODEL_FILE = 'trained_model.pth'

VOCAB_SIZE = 5000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
NUM_EPOCHS = 5
BATCH_SIZE = 32
SEQ_LENGTH = 100
LEARNING_RATE = 0.001
GRAD_CLIP = 5  


def download_and_extract_data() -> None:
    """Download and extract the dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EXTRACTED_FILE):
        logging.info("Data already downloaded.")
        return
    
    try:
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()
        compressed_file = os.path.join(DATA_DIR, 'dataset.tar.xz')
        
        with open(compressed_file, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc="Downloading"):
                if chunk:
                    f.write(chunk)
        
        with open(compressed_file, 'rb') as f:
            decompressed_data = pylzma.decompress(f.read())
        
        with open(EXTRACTED_FILE, 'wb') as f:
            f.write(decompressed_data)
        
        os.remove(compressed_file)
        logging.info("Data extraction complete.")
    except Exception as e:
        logging.error(f"Error during data extraction: {e}")
        raise


def train_tokenizer() -> Tokenizer:
    """Train or load a tokenizer."""
    if os.path.exists(TOKENIZER_FILE):
        logging.info("Tokenizer already exists. Loading...")
        return Tokenizer.from_file(TOKENIZER_FILE)
    
    logging.info("Training new tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2)
    
    try:
        with open(EXTRACTED_FILE, 'r', encoding='utf-8') as f:
            tokenizer.train_from_iterator(f, trainer=trainer)
        
        tokenizer.post_processor = processors.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(TOKENIZER_FILE)
        
        logging.info("Tokenizer training complete.")
        return tokenizer
    except Exception as e:
        logging.error(f"Error during tokenizer training: {e}")
        raise


class TextDataset(Dataset):
    """Custom Dataset for text data."""
    def __init__(self, text: str, tokenizer: Tokenizer, seq_length: int = SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(text).ids
        self.seq_length = seq_length
        self.num_samples = len(self.tokens) - self.seq_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx:idx + self.seq_length]
        y = self.tokens[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTMModel(nn.Module):
    """LSTM-based language model."""
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(device),
                torch.zeros(NUM_LAYERS, batch_size, HIDDEN_DIM).to(device))


def train_model(model: nn.Module, dataset: Dataset, num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE,
                learning_rate: float = LEARNING_RATE, grad_clip: float = GRAD_CLIP) -> None:
    """Train the model."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        hidden = model.init_hidden(batch_size)
        total_loss = 0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            hidden = tuple(h.detach() for h in hidden)

            model.zero_grad()
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), MODEL_FILE)
    logging.info("Model saved successfully!")


def load_model() -> nn.Module:
    """Load a trained model."""
    model = LSTMModel().to(device)
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        logging.info("Model loaded successfully.")
    return model


def generate_text(model: nn.Module, tokenizer: Tokenizer, start_text: str, length: int = 100,
                  temperature: float = 1.0) -> str:
    """Generate text using the trained model."""
    model.eval()
    tokens = tokenizer.encode(start_text).ids
    input_seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated = tokens

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            output_dist = torch.nn.functional.softmax(output[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(output_dist, 1).item()
            generated.append(next_token)
            input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)

    return tokenizer.decode(generated)


if __name__ == "__main__":
    try:
      
        download_and_extract_data()

    
        tokenizer = train_tokenizer()

    
        with open(EXTRACTED_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
        dataset = TextDataset(text, tokenizer)

        model = load_model()
        if not os.path.exists(MODEL_FILE):
            train_model(model, dataset)

    
        start_text = "Once upon a time"
        generated_text = generate_text(model, tokenizer, start_text, length=100, temperature=0.8)
        print(f"\nGenerated Text:\n{generated_text}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from src.data_processing import load_and_prepare_data, Vocabulary # Load your functions
from src.model import SmilesGenerator                           # Load your model
from tqdm import tqdm
import os

# --- 1. Configuration and Hyperparameters ---
# Define constants for file paths and training settings
DATA_PATH = 'data/SMILES_Big_Data_Set.csv'
MODEL_SAVE_PATH = 'models/best_smiles_generator.pt'

# Training Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Model Hyperparameters (Must match the setup in src/model.py)
EMBED_SIZE = 128
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT_RATE = 0.5


def train_model():
    # --- 2. Setup Device and Data ---
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data (from Phase 4)
    data, vocab = load_and_prepare_data(DATA_PATH)
    
    # Create PyTorch Dataset and DataLoader
    dataset = TensorDataset(torch.from_numpy(data))
    # DataLoader shuffles and batches the data
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = SmilesGenerator(
        vocab_size=vocab.vocab_size, 
        embed_size=EMBED_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # Loss: CrossEntropyLoss is perfect for multi-class classification (predicting the next token)
    # We ignore the PAD token (index 0) in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token_to_idx['<pad>']) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # --- 4. Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train() # Set model to training mode
        epoch_loss = 0
        
        # Use tqdm for a nice progress bar
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            # Input is the sequence batch [BATCH_SIZE, MAX_LEN]
            sequence_tensor = batch[0].to(device) 
            
            # Target is the next token in the sequence (shifted by one position)
            # The input is sequence_tensor[:, :-1]
            # The target is sequence_tensor[:, 1:]
            
            input_seq = sequence_tensor[:, :-1]
            target_seq = sequence_tensor[:, 1:]
            
            # Initialize hidden state for the start of the batch
            hidden = model.init_hidden(BATCH_SIZE, device) 

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            # logits: (BATCH_SIZE * (MAX_LEN-1), VOCAB_SIZE)
            logits, hidden = model(input_seq, hidden)
            
            # Calculate loss. Target must be reshaped to match logits shape.
            loss = criterion(logits, target_seq.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            
            # Clip gradients to prevent "exploding gradients" (optional but often helpful for RNNs)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

        # --- 5. Save the Best Model ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving new best model with loss: {best_loss:.4f}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("Training finished.")

if __name__ == '__main__':
    train_model()
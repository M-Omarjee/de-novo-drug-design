import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors
from src.data_processing import load_and_prepare_data, Vocabulary, SOS_TOKEN, EOS_TOKEN
from src.model import SmilesGenerator
import numpy as np

# --- 1. Configuration ---
DATA_PATH = 'data/SMILES_Big_Data_Set.csv'
MODEL_LOAD_PATH = 'models/best_smiles_generator.pt'

# Model Hyperparameters (Must match src/train.py)
EMBED_SIZE = 128
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT_RATE = 0.5
MAX_LEN = 150
TEMPERATURE = 0.8  # Controls randomness. Lower = more predictable, Higher = more diverse/junk.
NUM_TO_GENERATE = 1000


def generate_smiles_sequence(model, vocab, device):
    """Generates a single SMILES sequence token by token."""
    
    # 1. Start with the SOS token
    start_token_idx = vocab.token_to_idx[SOS_TOKEN]
    input_token = torch.tensor([[start_token_idx]], dtype=torch.long).to(device)
    
    # 2. Initialize hidden state
    hidden = model.init_hidden(batch_size=1, device=device)
    
    generated_indices = [start_token_idx]

    # 3. Generate tokens iteratively
    for _ in range(MAX_LEN):
        # Forward pass
        with torch.no_grad():
            # Pass only one token (the last generated one) at a time
            logits, hidden = model(input_token, hidden)
        
        # Apply temperature to logits for controlled sampling
        logits = logits.squeeze(0) / TEMPERATURE
        
        # Softmax and sample the next token
        probabilities = torch.softmax(logits, dim=-1)
        # Use multinomial for stochastic sampling
        next_token_idx = torch.multinomial(probabilities, 1).item()
        
        # Check for EOS token
        if next_token_idx == vocab.token_to_idx[EOS_TOKEN]:
            break
        
        # Append and set up for the next loop
        generated_indices.append(next_token_idx)
        input_token = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)

    # 4. Convert indices back to SMILES string, removing SOS/EOS
    smiles = "".join([vocab.idx_to_token[idx] for idx in generated_indices if idx not in [start_token_idx, vocab.token_to_idx[EOS_TOKEN]]])
    return smiles

def validate_and_score(smiles, valid_smiles_set):
    """Checks validity, novelty, and calculates a property (LogP)."""
    
    # Check RDKit Validity
    try:
        mol = Chem.MolFromSmiles(smiles)
        is_valid = mol is not None
    except Exception:
        is_valid = False

    if not is_valid:
        return False, False, None
        
    # Check Novelty (against the original training set)
    is_novel = smiles not in valid_smiles_set
    
    # Calculate a simple property (LogP for drug-likeness)
    logp = Descriptors.MolLogP(mol)
    
    return is_valid, is_novel, logp

def main_generate():
    device = torch.device("cpu") # Generating on CPU is typically fine
    print(f"Using device: {device}")

    # Load data to get the vocabulary and the training set for novelty check
    _, vocab = load_and_prepare_data(DATA_PATH, max_len=MAX_LEN)
    
    # CRITICAL: Create a set of all original SMILES strings for fast novelty checking
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    training_smiles = set(df['SMILES'].tolist())
    
    # Initialize the model and load the trained weights
    model = SmilesGenerator(
        vocab_size=vocab.vocab_size, 
        embed_size=EMBED_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    except FileNotFoundError:
        print(f"\nERROR: Model weights not found at {MODEL_LOAD_PATH}.")
        print("Please wait for training (src/train.py) to finish and save a checkpoint.")
        return

    model.eval() # Set model to evaluation mode

    generated_results = []
    
    print(f"\nStarting generation of {NUM_TO_GENERATE} molecules...")
    
    for i in range(NUM_TO_GENERATE):
        generated_smiles = generate_smiles_sequence(model, vocab, device)
        is_valid, is_novel, logp = validate_and_score(generated_smiles, training_smiles)

        if is_valid:
            generated_results.append({
                'smiles': generated_smiles,
                'valid': is_valid,
                'novel': is_novel,
                'LogP': logp
            })
            
    # --- 4. Report Results ---
    valid_molecules = [res for res in generated_results if res['valid']]
    novel_valid_molecules = [res for res in valid_molecules if res['novel']]
    
    validity_rate = len(valid_molecules) / NUM_TO_GENERATE * 100
    uniqueness_rate = len(set(m['smiles'] for m in generated_results)) / NUM_TO_GENERATE * 100
    novelty_rate = len(novel_valid_molecules) / len(valid_molecules) * 100 if len(valid_molecules) > 0 else 0

    print("\n--- GENERATION SUMMARY ---")
    print(f"Attempted to generate: {NUM_TO_GENERATE}")
    print(f"1. Validity Rate: {validity_rate:.2f}% ({len(valid_molecules)} valid)")
    print(f"2. Uniqueness Rate: {uniqueness_rate:.2f}%")
    print(f"3. Novelty Rate (among valid): {novelty_rate:.2f}% ({len(novel_valid_molecules)} novel)")
    
    print("\n--- Example Novel Valid Molecules ---")
    # Print the top 5 novel molecules
    for res in novel_valid_molecules[:5]:
        print(f"SMILES: {res['smiles']}, LogP: {res['LogP']:.2f}")

if __name__ == '__main__':
    main_generate()
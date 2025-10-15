import numpy as np
import pandas as pd # <-- Essential for reading the CSV file

# --- 1. Define Special Tokens and Tokenizer ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = 'G' 
EOS_TOKEN = 'E' 

def smiles_tokenizer(smiles):
    """
    Splits a SMILES string into a list of chemical tokens (simple character-level split).
    """
    if not isinstance(smiles, str):
        # Handle cases where a value might not be a valid string 
        return [] 
    return list(smiles)


# --- 2. Vocabulary Class (The Translator) ---
class Vocabulary:
    def __init__(self, smiles_list):
        # Filter out any non-string entries just to be safe
        self.smiles_list = [s for s in smiles_list if isinstance(s, str) and s.strip()] 
        
        self.token_to_idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
        self.idx_to_token = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN}
        self.build_vocab()

    def build_vocab(self):
        all_tokens = set()
        for smiles in self.smiles_list:
            tokens = smiles_tokenizer(smiles)
            all_tokens.update(tokens)
        
        for token in sorted(list(all_tokens)):
            if token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
        
        self.vocab_size = len(self.token_to_idx)

    def smiles_to_indices(self, smiles):
        """Converts a single SMILES string into a list of numerical indices."""
        tokens = smiles_tokenizer(smiles)
        indexed = [self.token_to_idx[SOS_TOKEN]] + \
                  [self.token_to_idx.get(token, self.token_to_idx[PAD_TOKEN]) for token in tokens] + \
                  [self.token_to_idx[EOS_TOKEN]]
        return indexed

# --- 3. Main Loader and Padder Function (CSV Handling) ---
def load_and_prepare_data(filepath, max_len=150):
    """
    Loads SMILES data from the CSV file, extracts the SMILES column,
    builds vocabulary, converts SMILES to indices, and pads them.
    """
    print("Loading raw SMILES data from CSV...")
    
    # Use pandas to read the CSV and extract the SMILES column
    df = pd.read_csv(filepath)
    # CRITICAL: We assume the SMILES column is named 'SMILES'
    smiles_list = df['SMILES'].tolist() 
    
    # Clean up the list
    smiles_list = [s.strip() for s in smiles_list if isinstance(s, str) and s.strip()]

    # 1. Build Vocabulary
    vocab = Vocabulary(smiles_list)
    print(f"Vocabulary built with {vocab.vocab_size} unique tokens from {len(smiles_list)} molecules.")

    # 2. Convert all SMILES to index sequences
    data_indices = [vocab.smiles_to_indices(s) for s in smiles_list]

    # 3. Pad sequences
    print(f"Padding sequences to length {max_len}...")
    padded_data = np.full((len(data_indices), max_len), vocab.token_to_idx[PAD_TOKEN], dtype=np.int64)
    
    for i, seq in enumerate(data_indices):
        length = min(len(seq), max_len)
        padded_data[i, :length] = np.array(seq[:length], dtype=np.int64)
    
    print("Data preparation complete.")
    return padded_data, vocab

# --- 4. Test Block ---
if __name__ == '__main__':
    # File path must match your data folder and file name
    FILE_PATH = 'data/SMILES_Big_Data_Set.csv' 
    try:
        data, vocab = load_and_prepare_data(FILE_PATH)
        print("\n--- TEST RESULTS ---")
        print(f"Total sequences ready for model: {data.shape[0]}")
        print(f"Max sequence length (padded): {data.shape[1]}")
        # print(f"First 5 padded sequences (numerical codes):\n {data[:5]}") # Uncomment if you want to see the numbers
        
    except FileNotFoundError:
        print(f"\nERROR: The file '{FILE_PATH}' was not found.")
        print("Please check the filename and ensure it is in the 'data/' folder.")
    except KeyError:
        print("\nERROR: Could not find the 'SMILES' column in the CSV file.")
        print("Please check the column name in your CSV is exactly 'SMILES'.")
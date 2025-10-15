import torch
import torch.nn as nn

class SmilesGenerator(nn.Module):
    """
    A Recurrent Neural Network (RNN) using LSTM layers for generating SMILES strings.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=3, dropout_rate=0.5):
        super(SmilesGenerator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # 1. Embedding Layer: Converts token IDs (integers) into dense vectors (embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. LSTM Layer: The core recurrent layer that processes sequences and maintains memory
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate)
        
        # 3. Output Layer: Maps the RNN output back to the size of the vocabulary
        #    This predicts the probability of the next token
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_tensor, hidden_state):
        """
        Processes one step of a sequence (a batch of single tokens).
        input_tensor: (batch_size, 1) or (batch_size, sequence_length)
        hidden_state: (h_n, c_n) containing the previous hidden state and cell state
        """
        # Embed the input tokens
        embedded = self.embedding(input_tensor) # Shape: (batch_size, seq_len, embed_size)
        
        # Pass through the LSTM
        # output: (batch_size, seq_len, hidden_size)
        output, hidden_state = self.rnn(embedded, hidden_state)
        
        # Apply dropout
        output = self.dropout(output)

        # Pass through the linear layer to get logits for the next token
        # Reshape output for the linear layer: (batch_size * seq_len, hidden_size)
        logits = self.output_layer(output.reshape(-1, self.hidden_size)) 

        # logits shape: (batch_size * seq_len, vocab_size)
        return logits, hidden_state

    def init_hidden(self, batch_size, device):
        """Initializes the hidden and cell states (h_0 and c_0) with zeros."""
        # The hidden and cell states need to be initialized on the same device as the model
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
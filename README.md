# De Novo Drug Design using a Recurrent Neural Network (RNN)

This project implements a Recurrent Neural Network (RNN), specifically a GRU, to generate novel small molecules in the form of SMILES strings. It is a proof-of-concept for generative chemistry, inspired by the objective of finding new molecules with desired properties.

The key steps in this project are:
1. **Training**: An RNN learns the statistical language of known drug-like molecules (SMILES).
2. **Generation**: The trained model generates entirely new, never-before-seen SMILES sequences.
3. **Evaluation**: Generated molecules are assessed for chemical validity and novelty.

---

## 💻 Setup and Installation (macOS/Apple Silicon)

Due to complex dependency requirements (PyTorch and RDKit), using the **Conda** environment manager is highly recommended.

### 1. Create and Activate the Conda Environment

Open your terminal and execute the following commands sequentially:

```bash
# 1. Create a new environment with Python 3.10
conda create -n molgen python=3.10

# 2. Activate the new environment
conda activate molgen
```
2. Install Core Dependencies (PyTorch and RDKit)

These packages require specific installation channels to resolve complex dependencies on Apple Silicon architecture.

```bash
# Install PyTorch (Official PyTorch channel for the CPU version)
conda install pytorch -c pytorch

# Install RDKit (Cheminformatics library from the conda-forge channel)
conda install -c conda-forge rdkit

# Fix NumPy Compatibility Issue
pip install "numpy<2"
```
3. Install Project Requirements

Install the remaining utility libraries from the requirements.txt file:
```bash
pip install -r requirements.txt
```

## 🏃 Project Execution
Ensure your Conda environment is active ((molgen) is visible in your terminal prompt) and you are in the project's root directory.

# 1. Train the Model

The training script reads the data/SMILES_Big_Data_Set.csv file, processes it, and trains the GRU model for 20 epochs. It saves the best model checkpoint to models/best_smiles_generator.pt.

Run the training script as a module to handle all imports correctly:
```bash
python -m src.train
```
Note: Training on a CPU is time-intensive (estimated 3-7 hours). You can stop the training early with Ctrl+C after a few epochs, and a valid checkpoint will likely be saved.

# 2. Generate and Evaluate Novel Molecules

Once training is complete, the src.generate script loads the best model checkpoint, generates 1,000 new SMILES strings, validates them using RDKit, and checks for novelty against the original dataset.

Run the generation script as a module:
```bash
python -m src.generate
```
The output will provide key metrics: Validity Rate, Uniqueness Rate, and Novelty Rate, demonstrating the generative capability of the model.

# 💊 De Novo Drug Design using a Recurrent Neural Network (RNN)

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
### 2. Install Core Dependencies (PyTorch and RDKit)

These packages require specific installation channels to resolve complex dependencies on Apple Silicon architecture.

```bash
# Install PyTorch (Official PyTorch channel for the CPU version)
conda install pytorch -c pytorch

# Install RDKit (Cheminformatics library from the conda-forge channel)
conda install -c conda-forge rdkit

# Fix NumPy Compatibility Issue
pip install "numpy<2"
```
### 3. Install Project Requirements

Install the remaining utility libraries from the requirements.txt file:
```bash
pip install -r requirements.txt
```

## 🏃 Project Execution
Ensure your Conda environment is active ((molgen) is visible in your terminal prompt) and you are in the project's root directory.

### 1. Train the Model

The training script reads the data/SMILES_Big_Data_Set.csv file, processes it, and trains the GRU model for 20 epochs. It saves the best model checkpoint to models/best_smiles_generator.pt.

Run the training script as a module to handle all imports correctly:
```bash
python -m src.train
```
Note: Training on a CPU is time-intensive (estimated 3-7 hours). You can stop the training early with Ctrl+C after a few epochs, and a valid checkpoint will likely be saved.

### 2. Generate and Evaluate Novel Molecules

Once training is complete, the src.generate script loads the best model checkpoint, generates 1,000 new SMILES strings, validates them using RDKit, and checks for novelty against the original dataset.

Run the generation script as a module:
```bash
python -m src.generate
```
The output will provide key metrics: Validity Rate, Uniqueness Rate, and Novelty Rate, demonstrating the generative capability of the model.

## 📈 Final Generation Results
The model was trained for the full 20 epochs, achieving a minimum loss of 0.4886. The final generation script demonstrated high quality, diversity, and novelty in the output.

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Attempted to generate** | 1,000 | Total number of novel molecules the AI attempted to write. |
| **1. Validity Rate** | **91.80%** | Percentage of generated SMILES that are chemically plausible structures (passed RDKit check). |
| **2. Uniqueness Rate** | **89.20%** | Percentage of the generated molecules that were unique (not duplicates). |
| **3. Novelty Rate** | **65.36%** | Percentage of the valid molecules that were brand new (not present in the original 16,087 molecule training set). |

Conclusion: The model successfully invented 600 novel, unique, and valid chemical structures, demonstrating a strong capability for de novo molecular design.

## 🏆 Final Candidate Selection and 3D Visualization
The virtual screening process identified a specific molecule with an optimal LogP score of 2.08 (target ideal is 2.0), which is critical for excellent oral bioavailability. This molecule represents the most promising candidate from the AI's generation.

| Property | Value |
| :-- | :-- |
| SMILES String | CCCCCNC(=O)COC(=O)c1ccccc1OC(C)=O |
| LogP Score | 2.08 (near-ideal for drug absorption) |

### 3D Render from Online Viewer

This image is a screenshot of the molecule's calculated 3D structure, viewed in a web component after rendering the coordinates. Its three-dimensional shape is crucial for predicting how it would interact with a biological target.

<div align="center">
<img src="renders/3D Render Mol File Screenshot.png" alt="3D Render of the Novel Candidate Molecule" width="450" />
</div>

<div align="center">
Special thanks are extended to the [**Kaggle dataset**](https://www.kaggle.com/datasets/yanmaksi/big-molecules-smiles-dataset) for the training data and to the [**MolViewer**](https://molstar.org/viewer/) for the clear 3D visualization.
</div>
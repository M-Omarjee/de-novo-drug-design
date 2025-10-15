import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToFile

# --- 1. Configuration ---
# PASTE YOUR CHOSEN SMILES STRING HERE
# This SMILES string (LogP 2.08) is the most 'drug-like' candidate.
SMILES_STRING = 'CCCCCNC(=O)COC(=O)c1ccccc1OC(C)=O' 
OUTPUT_FILE_NAME = 'novel_molecule_3d_render.png'

def generate_and_render_3d(smiles):
    """
    Generates a 3D conformation for the molecule and saves a 2D render.
    """
    print(f"Processing SMILES: {smiles}")

    # 1. Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("ERROR: Invalid SMILES string. Cannot generate molecule.")
        return

    # 2. Add explicit hydrogens and generate 3D coordinates (Conformation)
    # This step uses the distance geometry algorithm (EmbedMolecule)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates. Use Try to catch failures for complex molecules.
    try:
        # Generate 10 possible conformations and minimize them (UFF is a fast force field)
        AllChem.EmbedMolecule(mol, maxAttempts=5000)
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"WARNING: Could not generate optimal 3D coordinates. Using simplified geometry. Error: {e}")
    
    # 3. Save the 3D coordinates to a standard chemical file format (.mol)
    # This file can be opened in external 3D viewers (like PyMOL or Chimera)
    mol_with_conf = Chem.RemoveHs(mol) # Remove H for cleaner visualization in external tools
    mol_file_path = os.path.join('renders', 'novel_molecule.mol')
    
    # Ensure the renders directory exists
    os.makedirs('renders', exist_ok=True)
    
    Chem.MolToMolFile(mol_with_conf, mol_file_path)
    print(f"Successfully saved 3D data to: {mol_file_path}")

    # 4. Generate a high-quality 2D image render of the 3D structure
    image_path = os.path.join('renders', OUTPUT_FILE_NAME)
    
    try:
        # Use MolToFile to render a high-resolution image
        MolToFile(
            mol, 
            image_path, 
            size=(500, 500), 
            legend=f"Novel Molecule (LogP 2.08)\n{SMILES_STRING}",
            kekule=True # Draw aromatic rings clearly
        )
        print(f"Successfully saved 2D render to: {image_path}")
    except Exception as e:
        print(f"ERROR: Failed to save the image render. Error: {e}")


if __name__ == '__main__':
    generate_and_render_3d(SMILES_STRING)
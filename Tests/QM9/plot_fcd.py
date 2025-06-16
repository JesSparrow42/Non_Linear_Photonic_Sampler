#!/usr/bin/env python3
import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from fcd import get_fcd as calculate_fcd
from dataset import SparseMolecularDataset
from models import MolGANGenerator
from utils import sample_z

def decode_molecules(edge_logits, node_logits, dataset):
    """
    Convert generator outputs into SMILES via RDKit.

    Parameters:
        edge_logits   (np.ndarray): shape (B, N, N, n_bond_types)
        node_logits   (np.ndarray): shape (B, N, n_atom_types)
        dataset       (SparseMolecularDataset): to pull out atom_types

    Returns:
        List[str]: SMILES for each sample in the batch
    """
    # 1) discretize
    edge_idx = np.argmax(edge_logits, axis=-1)  # (B, N, N)
    node_idx = np.argmax(node_logits, axis=-1)  # (B, N)

    smiles = []
    atom_types = dataset.atom_types  # e.g. ['C','O','N',…]
    # map bond‐type index → RDKit BondType
    bond_types_list = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]

    for emat, nmat in zip(edge_idx, node_idx):
        mol = Chem.RWMol()
        idx_map = {}  # original index → rdkit index (or None for skipped atoms)

        # ─── add atoms (skip PAD=0 and hydrogen=1) ───
        for i, at in enumerate(nmat):
            Z = int(atom_types[int(at)])
            if Z in (0, 1):            # skip padding and explicit hydrogens
                idx_map[i] = None
                continue
            rd_idx = mol.AddAtom(Chem.Atom(Z))
            idx_map[i] = rd_idx

        # ─── add bonds only between kept atoms ───
        N = len(nmat)
        for i in range(N):
            for j in range(i + 1, N):
                if idx_map[i] is None or idx_map[j] is None:
                    continue
                b = int(emat[i, j])
                if 0 < b < len(bond_types_list):
                    mol.AddBond(idx_map[i], idx_map[j], bond_types_list[b])

        # Sanitize and convert to canonical SMILES; catch failures
        try:
            Chem.SanitizeMol(mol)
            # keep only the largest connected component
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if not frags:
                smiles.append("")
                continue
            main = max(frags, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(main)
            smiles.append(Chem.MolToSmiles(main, canonical=True))
        except Exception:
            smiles.append("")  # invalid molecule
    return smiles

def main():
    parser = argparse.ArgumentParser(description="Compute & plot FCD vs. epoch")
    parser.add_argument('--weights-dir', type=str, required=True,
                        help="Root folder containing baseline/, boson/, nonlinear/")
    parser.add_argument('--data-path',   type=str, required=True,
                        help="Dataset folder for reference SMILES")
    parser.add_argument('--subset',      type=float, default=1.0,
                        help="Fraction of reference dataset to use")
    parser.add_argument('--n-samples',   type=int,   default=1000,
                        help="Molecules to sample per checkpoint")
    parser.add_argument('--device',      type=str,   default='cpu')
    args = parser.parse_args()
    # Load QM9 reference dataset
    dataset = SparseMolecularDataset(args.data_path, subset=args.subset)

    # Build reference SMILES directly from the dense adjacency & feature tensors
    reference_smiles = decode_molecules(dataset.adjacency, dataset.node_features, dataset)

    branches = ['baseline', 'boson', 'nonlinear']
    fcd_scores = {b: {} for b in branches}

    for branch in branches:
        br_dir = os.path.join(args.weights_dir, branch)
        for fn in os.listdir(br_dir):
            # filenames are like "seed123_epoch10.pt"
            m = re.match(r'seed(\d+)_epoch(\d+)\.pt', fn)
            if not m: continue
            seed, epoch = int(m.group(1)), int(m.group(2))

            ckpt = torch.load(os.path.join(br_dir, fn),
                              map_location=args.device)
            # reconstruct generator
            kp = ckpt['args']
            key = 'base' if branch=='baseline' else branch
            z_dim      = kp[f'z_dim_{key}']
            hidden_dim = kp[f'hidden_dim_{key}']
            tau        = kp[f'tau_{key}']

            G = MolGANGenerator(z_dim,
                                dataset.node_features.shape[1],
                                dataset.node_features.shape[2],
                                dataset.adjacency.shape[3],
                                hidden_dim=hidden_dim,
                                tau=tau).to(args.device)
            G.load_state_dict(ckpt['G'])
            G.eval()

            # sample & decode
            with torch.no_grad():
                z = sample_z(args.n_samples, z_dim, args.device)
                edge_logits, node_logits = G(z, hard=True)
                smiles = decode_molecules(edge_logits.cpu().numpy(),
                                          node_logits.cpu().numpy(),
                                          dataset)

            # keep only valid SMILES; FCD requires at least two molecules per set
            valid_ref   = [s for s in reference_smiles if Chem.MolFromSmiles(s)]
            valid_batch = [s for s in smiles           if Chem.MolFromSmiles(s)]
            if len(valid_batch) < 2 or len(valid_ref) < 2:
                print(f"Skipping epoch {epoch} (not enough valid molecules)")
                continue

            fcd = calculate_fcd(valid_ref, valid_batch)
            fcd_scores[branch].setdefault(epoch, []).append(fcd)

    # plot mean±std
    epochs = sorted(next(iter(fcd_scores.values())).keys())
    plt.figure(figsize=(8,6))
    for branch in branches:
        means = [np.mean(fcd_scores[branch][e]) for e in epochs]
        stds  = [np.std (fcd_scores[branch][e]) for e in epochs]
        plt.plot(epochs, means, label=branch.capitalize())
        plt.fill_between(epochs,
                         np.array(means)-np.array(stds),
                         np.array(means)+np.array(stds),
                         alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel('Fréchet ChemNet Distance')
    plt.title('FCD vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
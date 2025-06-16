from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Synthetic Accessibility (SA) score moved to rdMolDescriptors in RDKit 2024.03.
# Support older RDKit versions gracefully.
try:
    from rdkit.Chem.rdMolDescriptors import CalcSyntheticAccessibilityScore as _calc_sa
except ImportError:  # fallback for older RDKit
    _calc_sa = None
    import warnings
    warnings.warn(
        "RDKit is missing `CalcSyntheticAccessibilityScore`; "
        "SA metric will be returned as zero.", RuntimeWarning)

class MolecularMetrics:
    """
    Computes validity, QED, logP and SA score, weighted-sum reward.
    """
    def __init__(self, weights=None):
        # QM9 atom‐type indices → atomic numbers  (C, N, O, F, H)
        self.atom_nums = [6, 7, 8, 9, 1]
        # e.g. {'validity':0.1,'qed':0.3,'logp':0.3,'sa':0.3}
        self.weights = weights or {'validity':0.1,'qed':0.3,'logp':0.3,'sa':0.3}

    def build_mol(self, adj, node_onehot):
        """
        adj: [N,N,R], node_onehot: [N,T]
        returns RDKit Mol or None
        """
        N, _, R = adj.shape
        type_idx = node_onehot.argmax(-1).tolist()          # index 0..4
        atom_z   = [self.atom_nums[i] if i < len(self.atom_nums) else 1
                    for i in type_idx]
        emap = {i: Chem.Atom(z) for i, z in enumerate(atom_z)}
        mol = Chem.RWMol()
        for i in range(N):
            mol.AddAtom(emap[i])
        for i in range(N):
            for j in range(i+1, N):
                e_t = adj[i,j].argmax().item()
                if e_t > 0:  # 0 = no bond
                    mol.AddBond(i, j, Chem.rdchem.BondType.values[e_t-1])
        try:
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def validity(self, mols):
        return np.mean([1 if m is not None else 0 for m in mols])

    def qed(self, mols):
        scores = []
        for m in mols:
            scores.append(QED.qed(m) if m is not None else 0.0)
        return np.mean(scores)

    def logp(self, mols):
        scores = []
        for m in mols:
            scores.append(MolLogP(m) if m is not None else -5.0)
        return np.mean(scores)

    def sa(self, mols):
        """
        Synthetic Accessibility (SA) score – negative values are better.
        Falls back to 0 if RDKit build lacks the function.
        """
        if _calc_sa is None:
            return 0.0
        scores = []
        for m in mols:
            scores.append(-_calc_sa(m) if m is not None else -10.0)
        return np.mean(scores)

    def compute(self, adjs, node_onehots):
        mols = [ self.build_mol(a, x) for a,x in zip(adjs, node_onehots) ]
        v = self.validity(mols)
        q = self.qed(mols)
        lp = self.logp(mols)
        sa = self.sa(mols)
        reward = (self.weights['validity']*v +
                  self.weights['qed']*q +
                  self.weights['logp']*lp +
                  self.weights['sa']*sa)
        return {'validity':v, 'qed':q, 'logp':lp, 'sa':sa, 'reward':reward}
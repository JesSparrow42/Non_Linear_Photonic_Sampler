def decode_molecules(edge_logits, node_logits, dataset):
    """
    Convert generator outputs (edge & node categorical logits) to SMILES strings.

    Node channel 0 is treated as PAD (no atom); any node whose predicted class
    maps to atomic number 0 is skipped.  Bonds are only added between real atoms.
    """
    edge_idx = np.argmax(edge_logits, axis=-1)   # (B,N,N)
    node_idx = np.argmax(node_logits, axis=-1)   # (B,N)

    bond_types = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    smiles = []
    atom_types = dataset.atom_types  # index → atomic number (0 = PAD)

    for emat, nmat in zip(edge_idx, node_idx):
        mol = Chem.RWMol()
        idx_map = {}  # original index → rdkit index or None

        # add atoms (skip PAD)
        for i, at in enumerate(nmat):
            Z = atom_types[int(at)]
            # Skip padding (Z=0) *and* explicit hydrogens (Z=1) so only heavy atoms remain
            if Z in (0, 1):
                idx_map[i] = None
                continue
            rd_idx = mol.AddAtom(Chem.Atom(int(Z)))
            idx_map[i] = rd_idx

        # add bonds
        N = len(nmat)
        for i in range(N):
            for j in range(i + 1, N):
                if idx_map[i] is None or idx_map[j] is None:
                    continue
                b = int(emat[i, j])
                if 0 < b < len(bond_types):
                    mol.AddBond(idx_map[i], idx_map[j], bond_types[b])

        try:
            Chem.SanitizeMol(mol)
            smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            smiles.append("")  # invalid molecule

    return smiles
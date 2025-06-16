#!/usr/bin/env python3
"""
Evaluate saved MolGAN-based generators on molecular metrics
(QED, logP, SA, validity) using RDKit.

Assumes checkpoints were saved by tests/QM9/train.py:
    weights_dir/
        baseline/   seed<N>_epoch<E>.pt
        boson/      seed<N>_epoch<E>.pt
        nonlinear/  seed<N>_epoch<E>.pt
"""
import argparse, os, re, json, torch, numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors, Descriptors
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

from dataset import SparseMolecularDataset
from models   import MolGANGenerator
from utils    import sample_z


# ───────────────────── helpers ─────────────────────
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
            # Keep only the largest connected component
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if not frags:
                smiles.append("")  # empty
                continue
            main = max(frags, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(main)
            smiles.append(Chem.MolToSmiles(main, canonical=True))
        except Exception:
            smiles.append("")  # invalid molecule

    return smiles


# Try to import the RDKit‐contrib synthetic‑accessibility scorer.
try:
    import sascorer            # comes with RDKit Contrib (rdkit/Contrib/SA_Score)
except ImportError:
    sascorer = None

def sa_score(mol):
    """
    Synthetic‑accessibility score (Ertl & Schuffenhauer, 2009).

    Falls back to NaN if the `sascorer` module is unavailable so that the
    evaluation script still runs without crashing.
    """
    if sascorer is None:
        return float("nan")
    return sascorer.calculateScore(mol)


def mol_metrics(smiles):
    """Return validity, QED, logP, SA for a list of SMILES."""
    vals, qed, logp, sa = [], [], [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        valid = mol is not None
        vals.append(valid)
        if valid:
            qed.append(QED.qed(mol))
            logp.append(Crippen.MolLogP(mol))
            sa.append(sa_score(mol))
    validity = np.mean(vals)
    qed_mean = np.mean(qed) if qed else 0.0
    logp_mean = np.mean(logp) if logp else 0.0
    sa_mean = np.nanmean(sa) if sa else float("nan")
    return dict(validity=validity, qed=qed_mean, logp=logp_mean, sa=sa_mean)


def load_generator(ckpt_path, dataset, device):
    """Reconstruct MolGANGenerator from checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    # Determine latent‑space variant tags used in training args
    # train.py defines:
    #   baseline → suffix "_base"
    #   boson    → suffix "_bos"
    #   nonlinear→ suffix "_nl"
    if "baseline" in ckpt_path:
        branch_tag = "baseline"
        suffix = "base"
    elif "boson" in ckpt_path:
        branch_tag = "boson"
        suffix = "bos"
    else:
        branch_tag = "nonlinear"
        suffix = "nl"

    # Retrieve hyper‑parameters with graceful fallback to global keys
    z_dim  = args.get(f"z_dim_{suffix}",  args.get("z_dim"))
    hidden = args.get(f"hidden_dim_{suffix}", args.get("hidden_dim"))
    tau    = args.get(f"tau_{suffix}",   args.get("tau"))
    if z_dim is None or hidden is None or tau is None:
        raise KeyError("Missing z_dim / hidden_dim / tau in checkpoint args")

    # Build generator
    G = MolGANGenerator(
        z_dim,
        dataset.node_features.shape[1],
        dataset.node_features.shape[2],
        dataset.adjacency.shape[3],
        hidden_dim=hidden,
        tau=tau
    ).to(device)
    G.load_state_dict(ckpt["G"], strict=False)
    G.eval()
    return G, z_dim


# ───────────────────── main ─────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True,
                    help="Root folder containing baseline/, boson/, nonlinear/")
    ap.add_argument("--data-path",   required=True,
                    help="QM9 dataset directory (same as during training)")
    ap.add_argument("--subset",      type=float, default=1.0,
                    help="Fraction of QM9 to load (for atom types)")
    ap.add_argument("--n-samples",   type=int,   default=1000,
                    help="Molecules to draw per checkpoint")
    ap.add_argument("--device",      type=str,   default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = SparseMolecularDataset(args.data_path, subset=args.subset)

    summary = []
    branch_samples = {"baseline": [], "boson": []}  # collect (smiles, qed) per branch
    top_candidates = []  # (smiles, qed)

    for branch in ("baseline", "boson", "nonlinear"):
        dir_ = os.path.join(args.weights_dir, branch)
        if not os.path.isdir(dir_):
            continue
        # Select only the highest‑epoch checkpoint per seed
        ckpts = {}
        for fn in os.listdir(dir_):
            m = re.match(r"seed(\d+)_epoch(\d+)\.pt", fn)
            if not m:
                continue
            seed, epoch = int(m.group(1)), int(m.group(2))
            if seed not in ckpts or epoch > ckpts[seed][1]:
                ckpts[seed] = (fn, epoch)
        # Iterate over selected checkpoints
        for seed, (fn, epoch) in ckpts.items():
            ckpt_path = os.path.join(dir_, fn)
            G, z_dim = load_generator(ckpt_path, ds, device)

            with torch.no_grad():
                z = sample_z(args.n_samples, z_dim, device)
                edge, node = G(z, hard=True)
                smiles = decode_molecules(edge.cpu().numpy(),
                                          node.cpu().numpy(), ds)

            # collect QED per valid molecule for global and per‑branch lists
            for s in smiles:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    q = QED.qed(mol)
                    top_candidates.append((s, q))
                    if branch in branch_samples:
                        branch_samples[branch].append((s, q))

            stats = mol_metrics(smiles)
            summary.append({
                "branch":  branch,
                "seed":    seed,
                "epoch":   epoch,
                **stats
            })
            print(f"[{branch}] seed {seed} epoch {epoch} → "
                  f"valid {stats['validity']:.3f}, "
                  f"QED {stats['qed']:.3f}, "
                  f"logP {stats['logp']:.3f}, "
                  f"SA {stats['sa']:.3f}")

    # Helper: pick up to k entries whose QED values differ by >tol
    def distinct_top(entries, k=5, tol=1e-4):
        unique = []
        for s, q in sorted(entries, key=lambda x: x[1], reverse=True):
            if all(abs(q - uq) > tol for _, uq in unique):
                unique.append((s, q))
            if len(unique) == k:
                break
        return unique

    # ─── Visualise 5 Gaussian (baseline) + 5 Boson molecules ───
    try:
        gauss_top = distinct_top(branch_samples["baseline"])
        boson_top = distinct_top(branch_samples["boson"])
        combined  = gauss_top + boson_top
        if combined:
            mols = [Chem.MolFromSmiles(s) for s, _ in combined]
            legends = ([f"Gaussian QED {q:.3f}" for _, q in gauss_top] +
                       [f"Boson   QED {q:.3f}"   for _, q in boson_top])
            grid = Draw.MolsToGridImage(
                mols, molsPerRow=5, subImgSize=(250, 250), legends=legends, useSVG=False
            )
            out_fig = os.path.join(args.weights_dir, "gaussian_vs_boson_top5.png")
            grid.save(out_fig)
            print(f"\nSaved Gaussian vs Boson figure → {out_fig}")
            plt.figure(figsize=(14, 6))
            plt.imshow(grid)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("Could not create Gaussian vs Boson image:", e)

    # ─── Visualise top‑10 molecules by QED ───
    if top_candidates:
        top_candidates.sort(key=lambda x: x[1], reverse=True)
        top10 = top_candidates[:10]
        mols    = [Chem.MolFromSmiles(s) for s, _ in top10]
        legends = [f"QED {q:.3f}" for _, q in top10]

        grid_img = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(200, 200),
            legends=legends,
            useSVG=False,
        )

        fig_path = os.path.join(args.weights_dir, "top10_qed.png")
        grid_img.save(fig_path)
        print(f"\nSaved image of top‑10 molecules → {fig_path}")

        # Display with matplotlib when possible
        try:
            plt.figure(figsize=(12, 5))
            plt.imshow(grid_img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    # ─── Report unique & valid molecule counts for Gaussian and Boson branches ───
    print("\nUnique / valid molecule statistics (all checkpoints combined):")
    for branch_lbl, branch_key in [("Gaussian", "baseline"), ("Boson", "boson")]:
        smiles_list = [s for s, _ in branch_samples.get(branch_key, [])]
        valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s)]
        unique_valid = set(valid_smiles)
        print(f"  {branch_lbl:8s}: valid = {len(valid_smiles):6d}, "
              f"unique valid = {len(unique_valid):6d}")

    # optional: write JSON
    out = os.path.join(args.weights_dir, "evaluation_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved full results → {out}")


if __name__ == "__main__":
    main()
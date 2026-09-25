"""Structural backbones for protein-ligand complexes.

    EquiformerV2   SE(3)-equivariant graph transformer over pocket + ligand atoms;
                   only the invariant (l = 0) channel of the final layer is used
    Uni-Mol        ligand representation from its SMILES / conformer

Complexes come from PDB, PDBbind or BioLiP; when no experimental structure
exists an AlphaFold3 model is used and its pLDDT / PAE are carried on the
evidence item.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.data.schema import Evidence

POCKET_RADIUS = 8.0          # Angstrom around any ligand atom
MAX_ATOMS = 600
ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Se", "Fe", "Zn", "Mg", "Ca")
ELEMENT_INDEX = {e.upper(): i + 1 for i, e in enumerate(ELEMENTS)}      # 0 = other


class EquiformerInvariant(nn.Module):
    """EquiformerV2 backbone returning per-atom invariant features."""

    def __init__(self, out_dim: int, num_layers: int = 6, lmax: int = 4, cutoff: float = 6.0
                 ) -> None:
        super().__init__()
        from fairchem.core.models.equiformer_v2 import EquiformerV2Backbone
        self.backbone = EquiformerV2Backbone(num_layers=num_layers, lmax_list=[lmax],
                                             mmax_list=[2], max_radius=cutoff,
                                             sphere_channels=out_dim, otf_graph=True,
                                             use_pbc=False, regress_forces=False)

    def forward(self, atom_types: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        from torch_geometric.data import Batch, Data
        data = Batch.from_data_list([Data(atomic_numbers=atom_types, pos=pos,
                                          natoms=torch.tensor([len(pos)]))])
        emb = self.backbone(data)["node_embedding"]            # SO3 embedding
        return emb.embedding[:, 0, :]                           # l = 0 coefficients


class UniMolLigand(nn.Module):
    """Uni-Mol molecular representation projected to a fixed width."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        from unimol_tools import UniMolRepr
        self.repr = UniMolRepr(data_type="molecule")
        self.proj = nn.Linear(512, out_dim)

    def forward(self, smiles: str) -> torch.Tensor:
        cls = self.repr.get_repr([smiles], return_atomic_reprs=False)["cls_repr"][0]
        return self.proj(torch.as_tensor(cls, dtype=self.proj.weight.dtype,
                                         device=self.proj.weight.device))


def build_equiformer(out_dim: int) -> nn.Module:
    return EquiformerInvariant(out_dim)


def build_unimol(out_dim: int) -> nn.Module:
    return UniMolLigand(out_dim)


def featurize_complex(e: Evidence, structure_dir: str = "data/structures"
                      ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
    """Pocket and ligand atoms of the complex referenced by `e.struct_id`."""
    from Bio.PDB import MMCIFParser, NeighborSearch

    s = MMCIFParser(QUIET=True).get_structure(e.struct_id, f"{structure_dir}/{e.struct_id}.cif")
    atoms = list(s.get_atoms())
    ligand = [a for a in atoms if a.get_parent().id[0].startswith("H_")]
    search = NeighborSearch([a for a in atoms if a not in ligand])
    pocket = {a for l in ligand for a in search.search(l.coord, POCKET_RADIUS)}
    selected = (ligand + sorted(pocket, key=lambda a: a.serial_number))[:MAX_ATOMS]
    types = torch.tensor([ELEMENT_INDEX.get(a.element.upper(), 0) for a in selected])
    pos = torch.tensor([a.coord for a in selected], dtype=torch.float)
    pos = pos - pos.mean(0, keepdim=True)
    return types, pos, {"smiles": e.ligand_smiles or ""}

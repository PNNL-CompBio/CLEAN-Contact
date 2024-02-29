from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.distance import squareform, pdist
import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile, get_structure
from tqdm import tqdm
import pandas as pd
from urllib.request import urlretrieve

def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])

def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts

def main(args):

    print(args)

    output_dir = Path('data/resnet_data')

    df = pd.read_csv(args.input, sep=args.csv_sep)
    pdb_filename_format = 'AF-{}-F1-model_v4.pdb'
    prot_ids = df[args.id_col].unique().tolist()
    error_ids = []

    model = torchvision.models.get_model(args.emb_model, weights=args.emb_weight)
    model.fc = nn.Identity()
    model = model.eval().to(args.device)

    for prot_id in tqdm(prot_ids):
        pdb = args.pdb_dir / pdb_filename_format.format(prot_id)
        if not pdb.exists():
            try:
                urlretrieve(
                    f'https://alphafold.ebi.ac.uk/files/{pdb_filename_format.format(prot_id)}',
                    pdb
                )
            except:
                error_ids.append(prot_id)
                continue

        structure = get_structure(PDBFile.read(pdb))[0]
        contact_map = contacts_from_pdb(structure)

        cmap1 = np.array([contact_map, contact_map, contact_map])

        with torch.no_grad():
            emb = model(torch.tensor(cmap1).unsqueeze(0).to(args.device, dtype=torch.float32)).cpu().squeeze(0)

        torch.save(emb, output_dir / f'{prot_id}.pt')

    with open('error_ids.txt', 'w') as f:
        f.write('\n'.join(error_ids))

def parse_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--id-col', type=str, default='Entry')
    parser.add_argument('--csv-sep', type=str, default='\t')
    parser.add_argument('--pdb-dir', type=Path, required=True)
    parser.add_argument('--emb-model', type=str, default='resnet50')
    parser.add_argument('--emb-weight', type=str, default='IMAGENET1K_V2')
    parser.add_argument('--device', type=str, default='cuda:1')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    main(args)

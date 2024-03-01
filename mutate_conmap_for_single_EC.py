import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from pathlib import Path

def read_masked_fasta(fasta_file):

    ret = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                id = line.strip()[1:]
                ret[id] = ''
            else:
                ret[id] += line.strip()

    return ret

def mask_and_embed_contact_map(contact_map, masked_fasta, model, device):

    masked_fasta = masked_fasta.replace('<mask>', '*')
    masked_contact_map = contact_map.copy()
    masked_indices = [i for i, c in enumerate(masked_fasta) if c == 'X']
    
    masked_contact_map[masked_indices, :] = 0
    masked_contact_map[:, masked_indices] = 0

    cmap1 = np.array([masked_contact_map, masked_contact_map, masked_contact_map])

    with torch.no_grad():
        emb = model(torch.tensor(cmap1).unsqueeze(0).to(device, dtype=torch.float32)).cpu().squeeze(0)

    return masked_contact_map, emb

def main(args):

    print(args)

    emb_dir = Path('data/resnet_data')
    conmap_dir = Path('data/contact_maps')

    fasta_dict = read_masked_fasta(args.fasta)
    prot_ids = list(fasta_dict.keys())

    model = torchvision.models.get_model(args.emb_model, weights=args.emb_weight)
    model.fc = nn.Identity()
    model = model.eval().to(args.device)

    for prot_id in tqdm(prot_ids):
        masked_fasta = fasta_dict[prot_id]
        ori_prot_id = prot_id.split('_')[0]
        contact_map = np.load(conmap_dir / f'{ori_prot_id}.npy')
        masked_contact_map, emb = mask_and_embed_contact_map(contact_map, masked_fasta, model, args.device)

        torch.save(emb, emb_dir / f'{prot_id}.pt')
        np.save(conmap_dir / f'{prot_id}.npy', masked_contact_map)


def parse_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=Path, required=True, help='Mutated fasta file')
    parser.add_argument('--emb-model', type=str, default='resnet50')
    parser.add_argument('--emb-weight', type=str, default='IMAGENET1K_V2')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    main(args)
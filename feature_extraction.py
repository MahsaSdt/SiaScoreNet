import pandas as pd
import requests as re
import time
from tqdm import tqdm
from io import StringIO
import torch
from esm import pretrained
import esm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

input_file = args.input
output_file = args.output

df = pd.read_csv(input_file)
df = df[df['peptide'].str.len().between(8, 14)].reset_index(drop=True)

def get_score(peptides, HLA, method, col_idx):
    url = "https://api-nextgen-tools.iedb.org/api/v1/pipeline"
    payload = {
        "pipeline_id": "",
        "pipeline_title": "",
        "email": "",
        "run_stage_range": [1, 1],
        "stages": [{
            "stage_display_name": "T-Cell Prediction",
            "stage_number": 1,
            "stage_type": "prediction",
            "tool_group": "mhci",
            "input_sequence_text": "\n".join([f">{p}\n{p}" for p in peptides]),
            "input_parameters": {
                "alleles": HLA,
                "peptide_length_range": None,
                "predictors": [{"type": "binding", "method": method}]
            },
            "table_state": {"columns": {}}
        }]
    }
    try:
        res = re.post(url, json=payload).json()
        if "results_uri" not in res:
            return [None] * len(peptides)
        while True:
            time.sleep(1)
            r = re.get(res['results_uri']).json()
            if r["status"] == "done":
                break
        table = r['data']['results'][0]['table_data']
        return [dict((row[1], row[col_idx]) for row in table).get(p, None) for p in peptides]
    except:
        return [None] * len(peptides)

def get_score2(peptides, HLA, method, col_idx):
    url = "http://tools-cluster-interface.iedb.org/tools_api/mhci/"
    payload = {
        "method": method,
        "allele": ",".join([HLA] * len(peptides)),
        "sequence_text": "\n".join([f">{p}\n{p}" for p in peptides]),
        "length": ",".join([str(len(p)) for p in peptides]),
        "species": ",".join(["human"] * len(peptides))
    }
    try:
        r = re.post(url, data=payload)
        df_result = pd.read_csv(StringIO(r.text), sep="\t")
        if "peptide" not in df_result.columns:
            return [None] * len(peptides)
        return [dict(zip(df_result["peptide"], df_result.iloc[:, col_idx])).get(p, None) for p in peptides]
    except:
        return [None] * len(peptides)


methods_api = {
    'ann': 8, 'consensus': 8, 'netmhcpan_ba': 10,
    'netmhcpan_el': 10, 'smm': 8, 'smmpmbec': 8
}
methods_web = {
    'pickpocket': 6, 'netmhccons': 6, 'netmhcstabpan': 6
}

for m in list(methods_api.keys()) + list(methods_web.keys()):
    df[f'score_{m}'] = None

for method, idx in tqdm(methods_api.items(), desc="API methods"):
    for HLA in df['HLA'].unique():
        peps = df[df['HLA'] == HLA]['peptide'].tolist()
        scores = get_score(peps, HLA, method, idx)
        df.loc[df['HLA'] == HLA, f'score_{method}'] = scores

for method, idx in tqdm(methods_web.items(), desc="Web methods"):
    for HLA in df[['HLA'] == HLA]['peptide'].tolist():
        scores = get_score2(peps, HLA, method, idx)
        df.loc[df['HLA'] == HLA, f'score_{method}'] = scores

hla_map = pd.read_csv('MHC_pseudo.dat', sep='\s+', header=None)
hla_map[0] = [''.join([h[:5], '*', h[5:]]) for h in hla_map[0]]
hla_map[0] = [h if ':' in h else h[:8] + ':' + h[8:] for h in hla_map[0]]
hla_dict = dict(zip(hla_map[0], hla_map[1]))
df['HLA_sequence'] = df['HLA'].map(hla_dict)

esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

def get_esm_vector(seq):
    _, _, tokens = batch_converter([(seq, seq)])
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[6])
    return out["representations"][6].mean(1).squeeze().numpy()

def extract_vectors(row):
    try:
        hla_vec = get_esm_vector(row['HLA_sequence'])
        pep_vec = get_esm_vector(row['peptide'])
        return pd.Series(hla_vec.tolist() + pep_vec.tolist())
    except:
        return pd.Series([None]*640)

esm_cols = [f'HLA_ESM_{i}' for i in range(1, 321)] + [f'Peptide_ESM_{i}' for i in range(1, 321)]
df[esm_cols] = df.apply(extract_vectors, axis=1)

df.to_csv(output_file, index=False)


from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator, rdmolops, DataStructs
import pandas as pd
import re
from formula_validation.Formula import Formula
import argparse
import numpy as np
from rdkit.Chem.MolStandardize import rdMolStandardize
import base64
import math
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def tanimoto_match_using_stored_morgan(df, target_mol, tanimoto_threshold: float):
    """
    Requires columns: ['Smiles','inchikey_first_block','fp_morgan','fp_morgan_popcnt'].
    Assumes fp_morgan is a BLOB of bytes produced by DataStructs.BitVectToBinaryText.
    """

    # 1) Build the query Morgan FP once (same params you stored)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    qfp = gen.GetFingerprint(target_mol)
    qa = int(qfp.GetNumOnBits())

    # 2) Optional necessary popcount bounds to prune (no false negatives)
    t = float(tanimoto_threshold)
    bmin = int(np.floor(t * qa))        # necessary: b >= t * a
    bmax = int(np.ceil(qa / max(t, 1e-9)))  # necessary: b <= a / t

    # 3) Keep rows with a stored Morgan FP and plausible popcount
    cand = df.loc[
        df["fp_morgan"].notna()
        & df["fp_morgan_popcnt"].notna()
        & (df["fp_morgan_popcnt"].astype(int) >= bmin)
        & (df["fp_morgan_popcnt"].astype(int) <= bmax)
    ].copy()

    if cand.empty:
        return df.iloc[0:0].copy()

    # 4) De-dup (your original logic) then take one FP per unique SMILES
    uniq_smiles = (
        cand.groupby("inchikey_first_block")["Smiles"]
        .first().dropna().unique().tolist()
    )

    # Build a lookup: SMILES -> bytes (ensure bytes, not str)
    # (We select first non-null fp_morgan per SMILES)
    sub = cand[["Smiles", "fp_morgan"]].dropna().drop_duplicates("Smiles")
    def _as_bytes(v):
        return v.encode("latin1") if isinstance(v, str) else (bytes(v) if not isinstance(v, (bytes, bytearray)) else v)
    fp_bytes_map = {row.Smiles: _as_bytes(row.fp_morgan) for row in sub.itertuples(index=False)}

    # 5) Confirm by Tanimoto using stored bytes (no MolFromSmiles here)
    hits = []
    for s in tqdm(uniq_smiles, desc=f"Tanimoto ≥ {t:.2f} (stored Morgan)"):
        b = fp_bytes_map.get(s)
        if not b:
            continue
        tfp = DataStructs.CreateFromBinaryText(b)
        if DataStructs.TanimotoSimilarity(qfp, tfp) >= t:
            hits.append(s)

    # 6) Return all rows matching those SMILES
    if not hits:
        return df.iloc[0:0].copy()
    return df[df["Smiles"].isin(set(hits))]

#Credit Michael Strobel with changes
def neutralize_atoms(smiles):
    """This function takes in a smiles and verifies that it does not have a charge that can be
    neutralized by protonation or deprotonation. If it does, it will neutralize the charge and return the mol.
    Code Source: http://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules
    Returns:
        Union[int,bool,int,rdkit.Chem.rdchem.Mol]: The number of charges removed, whether the resulting mol is 
        both pos and neg charged, the sum of the charges, and the neutralized mol.
    """    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    num_removed_charges = len(at_matches_list)
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    pos = False
    neg = False
    sum_of_charges = 0
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge < 0:
            neg = True
        if charge > 0:
             pos = True
        sum_of_charges += charge
    if pos and neg:
        pos_and_neg =True
    else:
        pos_and_neg = False
    
    return Chem.MolToSmiles(mol)

def tautomerize_smiles(smiles):
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = 100000
    params.maxTransforms = 100000
    te = rdMolStandardize.TautomerEnumerator(params)
    mol = Chem.MolFromSmiles(smiles)
    mol = te.Canonicalize(mol)
    standard_smiles = Chem.MolToSmiles(mol)
    del mol
    return standard_smiles

def detect_smiles_or_smarts(s: str) -> str:
    """
    Heuristically detect whether a string is a SMILES, SMARTS, or Invalid.
    """

    mol_smiles = Chem.MolFromSmiles(s)
    mol_smarts = Chem.MolFromSmarts(s)

    # SMARTS-specific patterns not common in SMILES
    smarts_tokens = [
        r"\[\#\d+",       # atom class: [#6]
        r";", r",",       # logical OR/AND
        r"&", r"!",       # NOT, AND
        r"\$\(.*\)",      # recursive SMARTS
        r"D\d", r"H\d",   # degree or hydrogen count
        r"R\d?",          # ring membership (rare in SMILES)
        r"X\d",           # connectivity
        r"a", r"A",       # aromatic / aliphatic queries
    ]

    if mol_smiles and not mol_smarts:
        return "smiles"
    elif mol_smiles and mol_smarts:
        smarts_pattern = re.compile("|".join(smarts_tokens))
        if smarts_pattern.search(s):
            return "smarts"
        else:
            return "smiles"
    elif mol_smarts:
        return "smarts"
    else:
        return "Invalid"

def fetch_and_match_smiles(df, target_smiles, match_type='exact', smiles_name='only', smiles_type='unknown', formula_base='any', element_diff='any', tanimoto_threshold=0.8, max_by_grp = None, max_overall = None):

    
    #make all column names lower case
    # df.columns = df.columns.str.lower()

    print(f"Detected SMILES type: {smiles_type}")
    # Check and convert the target SMILES to a molecule object
    if smiles_type == 'smiles':
        target_mol = Chem.MolFromSmiles(target_smiles)
    elif smiles_type == 'smarts':
        match_type = 'substructure'
        target_mol = Chem.MolFromSmarts(target_smiles)

    if target_mol is None:
        raise ValueError(f"Invalid SMILES: {target_smiles}")
    target_inchi_key = Chem.MolToInchiKey(target_mol).split('-')[0]

    # Parse formula_base and element_diff
    if formula_base != 'any':
        formula_base = Formula.formula_from_str(formula_base)
    if element_diff != 'any':
        element_diff = Formula.formula_from_str(element_diff)

    df['inchikey_first_block'] = df['InChIKey_smiles'].apply(
    lambda inchi: str(inchi).split('-')[0] if pd.notnull(inchi) else None
)

    if match_type == 'exact':
        # Perform exact match based on InChIKey
        df_matched = df[df['inchikey_first_block'] == target_inchi_key]
    elif match_type == 'substructure':

        # 1) Build query fingerprint once from your existing target_mol
        qfp = rdmolops.PatternFingerprint(target_mol, fpSize=2048)

        # 2) Prescreen via fingerprint subset (no false negatives)
        keep = []
        for raw in df["fp_pattern"].values:
            if raw is None:
                keep.append(False)
                continue
            # accept ExplicitBitVect or bytes/memoryview
            if isinstance(raw, ExplicitBitVect):
                bv = raw
            elif isinstance(raw, (bytes, bytearray, memoryview)):
                try:
                    bv = DataStructs.CreateFromBinaryText(bytes(raw))
                except Exception:
                    keep.append(False)
                    continue
            else:
                keep.append(False)
                continue

            keep.append(DataStructs.AllProbeBitsMatch(qfp, bv))

        df_screened = df.loc[np.array(keep)]
        if df_screened.empty:
            df_matched = df_screened
        else:
            # 3) De-dup by first InChIKey block (as before) but only among screened rows
            unique_smiles = (
                df_screened.groupby('inchikey_first_block')['Smiles']
                .first().dropna().unique().tolist()
            )

            # 4) Confirm true hits with RDKit only on screened uniques
            smiles_to_mol = {}
            for s in unique_smiles:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    smiles_to_mol[s] = m

            matching_smiles = []
            for s, m in tqdm(smiles_to_mol.items(),
                            desc=f"Confirming substructure on {len(smiles_to_mol)} candidates"):
                if m.HasSubstructMatch(target_mol):
                    matching_smiles.append(s)

            df_matched = df[df['Smiles'].isin(matching_smiles)]



    elif match_type == 'tanimoto':
        df_matched = tanimoto_match_using_stored_morgan(df, target_mol, tanimoto_threshold)

    # If no matching SMILES were found, return an empty list
    if df_matched.empty:
        print("No matching structures found.")
        return []

    df_matched = df_matched.copy()


    if max_by_grp is not None:

        df_matched[['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']] = df_matched[
            ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership']
        ].fillna('unknown')

        # Group by the required columns and limit to at most 8 rows per group
        df_matched['row_num'] = df_matched.groupby(
            ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'GNPS_library_membership', 'InChIKey_smiles']
        ).cumcount() + 1

        # Keep only the first 8 rows per group
        df_limited = df_matched[df_matched['row_num'] <= max_by_grp]
        
        if len(df_limited) > max_overall:
            df_matched['row_num'] = df_matched.groupby(
                ['collision_energy', 'Adduct', 'msManufacturer', 'msMassAnalyzer', 'InChIKey_smiles']
            ).cumcount() + 1

            # Keep only the first 8 rows per group
            df_limited = df_matched[df_matched['row_num'] <= max_by_grp]

        if len(df_limited) > max_overall:
            df_matched['row_num'] = df_matched.groupby(
                ['collision_energy', 'Adduct', 'InChIKey_smiles']
            ).cumcount() + 1

            # Keep only the first 8 rows per group
            df_limited = df_matched[df_matched['row_num'] <= max_by_grp]

        if len(df_limited) > max_overall:
            df_matched['row_num'] = df_matched.groupby(
                ['Adduct', 'InChIKey_smiles']
            ).cumcount() + 1

            # Keep only the first 8 rows per group
            df_limited = df_matched[df_matched['row_num'] <= max_by_grp]

        if len(df_limited) > max_overall:
            df_matched['row_num'] = df_matched.groupby(
                ['InChIKey_smiles']
            ).cumcount() + 1

            # Keep only the first 8 rows per group
            df_limited = df_matched[df_matched['row_num'] <= max_by_grp]

        if len(df_limited) > max_overall:
            print(f"Warning: More than {max_overall} matching structures found. Limiting to {max_overall}.")

            df_limited = df_limited.head(max_overall)

        print(f"Found {len(df_limited)} matching structures after limiting by group.")
    
    else:
        df_limited = df_matched

    df_limited.loc[:, 'smiles_name'] = smiles_name


    #rename spectrum_id to query_spectrum_id
    

    #remove entries where query id does not start on CCMS or MSBNK
    #df_limited = df_limited[df_limited['query_spectrum_id'].str.startswith(('CCMS', 'MSBNK'))]

    #reindex
    df_limited.reset_index(drop=True, inplace=True)

    print(f"Returning {len(df_limited)} unique spectrum_ids for {smiles_name}.")

    # Return the unique spectrum_ids
    return df_limited

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match SMILES')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--target', help='Target SMILES', default = '') 
    parser.add_argument('--match_type', help='Matching type: exact or substructure', default='exact')
    parser.add_argument('--smiles_name', help='Name of the matched SMILES', default='only')
    parser.add_argument('--smiles_type', help='Type of SMILES: smiles or smarts', default='smiles')
    parser.add_argument('--formula_base', help='Formula base for matching', default='any')
    parser.add_argument('--element_diff', help='Element difference for matching', default='any')
    parser.add_argument('--max_by_grp', help='Maximum number of rows to keep per group', default=8, type=int)
    parser.add_argument('--test', help='test', default=False, type=bool)

    args = parser.parse_args()


    if args.test:
        #read columns as string
        df = pd.read_csv(args.input, dtype=str)

        # test smiles example

        target_smiles = 'CN1[C@@H]2CC(C[C@H]1[C@H]3[C@@H]2O3)OC(=O)[C@H](CO)C4=CC=CC=C4' # phe-ca
        max_by_grp = 2

        matched_smiles = fetch_and_match_smiles(
            df = df, target_smiles = target_smiles, match_type = 'exact', max_by_grp = 2
        )

        print(matched_smiles)

        #read columns as string
        df = pd.read_csv(args.input, dtype=str)
        target_smiles = '[#6:1]-[#7:2]1-[#6@@H:3]2-[#6:4]-[#6:5](-[#8:6])-[#6:7]-[#6@H:8]-1-[#6@H:9]1-[#6@@H:10]-2-[#8:11]-1' # phe-ca
        max_by_grp = 2

        matched_smiles = fetch_and_match_smiles(
            df = df, target_smiles = target_smiles, match_type = 'substructure', max_by_grp = 2, smiles_type = 'smarts'
        )

        print(matched_smiles)


        exit



from rdkit import Chem, DataStructs
from tqdm import tqdm
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
import pandas as pd
import re
from formula_validation.Formula import Formula
import argparse

def detect_smiles_or_smarts(s):
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
        # Perform substructure matching
        unique_smiles = df.groupby('inchikey_first_block')['Smiles'].first().dropna().reset_index()
        unique_smiles = unique_smiles['Smiles'].dropna().unique()

        smiles_to_mol = {smiles: Chem.MolFromSmiles(smiles) for smiles in unique_smiles if Chem.MolFromSmiles(smiles) is not None}

        matching_smiles = []
        for smiles, mol in tqdm(smiles_to_mol.items(), desc="Substructure Matching", total=len(smiles_to_mol)):
            if mol.HasSubstructMatch(target_mol):
                # Formula difference matching
                if formula_base != 'any':
                    formula_candidate = rdMolDescriptors.CalcMolFormula(mol)
                    formula_candidate = Formula.formula_from_str(formula_candidate)

                    try:
                        formula_diff_here = formula_candidate - formula_base
                    except:
                        continue

                    diff_comparison = None
                    try:
                        diff_comparison = formula_diff_here - element_diff
                    except:
                        try:
                            diff_comparison = element_diff - formula_diff_here
                        except:
                            diff_comparison = None

                    match_is = (diff_comparison is None) or (set(re.sub(r'\d', '', str(diff_comparison))) == {"H"})

                    if not match_is:
                        continue

                matching_smiles.append(smiles)

        df_matched = df[df['Smiles'].isin(matching_smiles)]
    elif match_type == 'tanimoto':
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        target_fp = generator.GetFingerprint(target_mol)

        unique_smiles = df.groupby('inchikey_first_block')['Smiles'].first().dropna().reset_index()
        unique_smiles = unique_smiles['Smiles'].dropna().unique()

        matching_smiles = []
        for smiles in tqdm(unique_smiles, desc="Tanimoto Matching"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = generator.GetFingerprint(mol)
            sim = DataStructs.TanimotoSimilarity(target_fp, fp)
            if sim >= tanimoto_threshold:
                matching_smiles.append(smiles)

        df_matched = df[df['Smiles'].isin(matching_smiles)]

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



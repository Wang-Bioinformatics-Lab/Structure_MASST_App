import pandas as pd
import argparse
import os
import json
import time
import requests
from tqdm import tqdm
import sys


HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GNPSDataPackage'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


from gnpsdata import fasst



def query_fasst_usi(status, usi, analog=False, precursor_mz_tol=0.05,
                    matching_peaks=6, modimass=None, elimination=False, addition=False):

   
    try:
        modimass_val = float(modimass)
    except (TypeError, ValueError):
        modimass_val = None

    try:
        response = fasst.get_results(status, host="https://api.fasst.gnps2.org", blocking=True)
        

        response_list = response['results']

        
        if len(response_list) > 0:
            df = pd.DataFrame(response_list)

            if analog == False:
                df = df[df['Delta Mass'].abs() <= precursor_mz_tol]
                print(f"Delta Mass filter applied: {precursor_mz_tol}")

            elif analog == True:
                df['Delta Mass'] = df['Delta Mass'].astype(float)
                df['Delta Mass'] = df['Delta Mass'] * -1
                df = df[(df['Delta Mass'].abs() >= 5) | (df['Delta Mass'].abs() <= precursor_mz_tol)]
                df.loc[df['Delta Mass'].abs() <= precursor_mz_tol, 'Modified'] = 'no'
                df.loc[df['Delta Mass'] > precursor_mz_tol, 'Modified'] = 'addition'
                df.loc[df['Delta Mass'] < -precursor_mz_tol, 'Modified'] = 'elimination'

                # if delta mass is below 1 set it to 0
                df.loc[df['Delta Mass'].abs() < 1, 'Delta Mass'] = 0.0

                if modimass_val is not None:
                    df = df[
                        (df['Delta Mass'].abs() <= precursor_mz_tol) |
                        ((df['Delta Mass'].abs() - modimass_val).abs() <= precursor_mz_tol)
                    ]

                if elimination and not addition:
                    df = df[(df['Delta Mass'].abs() <= precursor_mz_tol) | (df['Delta Mass'] < 0)]

                if addition and not elimination:
                    df = df[(df['Delta Mass'].abs() <= precursor_mz_tol) | (df['Delta Mass'] > 0)]

            df = df[df['Matching Peaks'] >= matching_peaks]
            df['query_spectrum_id'] = usi
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Failed at retrieving {status} with usi {usi}")
        print(f"Error: {e}")
        return pd.DataFrame()







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query FASST USI')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('--database', help='Database to query', default='metabolomicspanrepo_index_latest')
    parser.add_argument('--analog', help='Analog search', default=False, type=bool)
    parser.add_argument('--precursor_mz_tol', help='Precursor m/z tolerance', default=0.05, type=float)
    parser.add_argument('--fragment_mz_tol', help='Fragment m/z tolerance', default=0.05, type=float)
    parser.add_argument('--min_cos', help='Minimum cosine score', default=0.7, type=float)
    parser.add_argument('--cache', help='Use cache', default="Yes")
    parser.add_argument('--test', help='test', default=False, type=bool)
    args = parser.parse_args()



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
                    matching_peaks=6, modimass=None, elimination=False, addition=False, log_output=None):

    #print(f"Querying FASST for USI {usi} with status {status} and parameters: analog={analog}, precursor_mz_tol={precursor_mz_tol}, matching_peaks={matching_peaks}, modimass={modimass}, elimination={elimination}, addition={addition}")
    try:
        modimass_val = float(modimass)
    except (TypeError, ValueError):
        modimass_val = None

    try:
        print(f"Submitting FASST query for USI {usi} with status {status}")
        
        # minimal NOT_FOUND grace retries
        response = None
        for _ in range(10):  # ~10s grace window
            response = fasst.get_results(status, blocking=True)  # pass the SAME host you submitted to
            if response.get("status") == "NOT_FOUND" and response.get("error") == "Invalid Task ID":
                time.sleep(1)
                continue
            elif 'error' in response:
                if log_output:
                    with open(os.path.join(log_output, f"{usi}_error.log"), "w") as f:
                        json.dump(response, f)

                raise RuntimeError(response['error'])
            break

        print(f"Response keys: {response.keys()}")

        # minimal safety: if no results, bail out (don’t crash)
        if "results" not in response:
            print(f"FASST returned no results payload: {response}")
            # return empty dataframe / None / raise — whatever your function expects
            raise RuntimeError(response.get("error", "No results in response"))

        response_list = response["results"]

        
        if len(response_list) > 0:
            df = pd.DataFrame(response_list)

            # If column "Charge" present make sure it is integeter and absolute value <= 1
            if 'Charge' in df.columns:
                df['Charge'] = pd.to_numeric(df['Charge'], errors='coerce').fillna(0).astype(int)
                df = df[df['Charge'].abs() <= 1]

            if analog == False:
                df = df[df['Delta Mass'].abs() <= precursor_mz_tol]
                print(f"Delta Mass filter applied: {precursor_mz_tol}")

            elif analog == True:
                #if in the USI column the first value ends on ".0" remove the last two characters from each entry
                if df["USI"].iloc[0].endswith(".0"):
                    df["USI"] = df["USI"].str[:-2]

                df['Delta Mass'] = df['Delta Mass'].astype(float)
                df['Delta Mass'] = df['Delta Mass'] * -1
                df['Unit Delta Mass'] = df['Delta Mass'].round(0).astype(int)
                df = df[(df['Delta Mass'].abs() >= 0.5) | (df['Delta Mass'].abs() <= precursor_mz_tol)]
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



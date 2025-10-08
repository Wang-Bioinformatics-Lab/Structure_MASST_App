import os
import json
import time
import requests
from tqdm import tqdm
import sys
import pandas as pd

HERE = os.path.dirname(__file__)  
PKG_PATH = os.path.abspath(os.path.join(HERE, '..', 'external', 'GNPSDataPackage'))

if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


from gnpsdata import fasst

FASST_API_SERVER_URL = "https://api.fasst.gnps2.org"





def get_results(query_parameters_dictionary, host="https://api.fasst.gnps2.org", blocking=True):
    task_id = query_parameters_dictionary["task_id"]
    
    retries_max = 20
    current_retries = 0
    while True:
        print("WAITING FOR RESULTS", current_retries, task_id)
        
        r = requests.get(os.path.join(host, "search/result/{}".format(task_id)), timeout=30)

        try:
            r.raise_for_status()
        except KeyboardInterrupt:
            raise
        except:
            # if we are not blocking, we just return the status
            if blocking is False:
                return "PENDING"
            
            time.sleep(1)
            current_retries += 1
            

            continue


        # checking if the results are ready
        if "status" in r.json() and r.json()["status"] == "PENDING":
            # if we are not blocking, we just return the status
            if blocking is False:
                return "PENDING"
            
            time.sleep(1)
            current_retries += 1

            if current_retries >= retries_max:
                raise Exception("Timeout waiting for results from FASST API")
            
            continue

        results_dict = r.json()
    
        return results_dict



def test_fasst_api_search_nonblocking():

    usi = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005435899"

    status_results_list = []
    for i in range(0, 1):
        print("submitted", i)
        results = fasst.query_fasst_api_usi(usi, "metabolomicspanrepo_index_nightly", host=FASST_API_SERVER_URL, analog=False, \
                    precursor_mz_tol=0.05, fragment_mz_tol=0.05, min_cos=0.7, cache="No", blocking=False)
        
        status_results_list.append(results)

    # lets now wait for all the results to be ready
    for status in status_results_list:
        results = get_results(status)
        response_list = results['results']
        len_results = len(response_list)

    return len_results



if __name__ == "__main__":
    results = test_fasst_api_search_nonblocking()
    print(results)
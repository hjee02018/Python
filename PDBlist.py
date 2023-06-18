import requests
import urllib.request
import concurrent.futures

pdb_ids = []
url = 'ftp://ftp.wwpdb.org/pub/pdb/derived_data/index/entries.idx'
save_path = 'pdb_ids.txt'
urllib.request.urlretrieve(url, save_path)

with open(save_path) as f:
    for i, line in enumerate(f):
        if i >= 3:
            pdb_ids.append(line[:4])

def get_pdb_info(pdb_id):
    with open('small_polypeptide_pdbs_2.txt', 'a') as f:
        #for pdb_id in pdb_ids:
            url = f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id}'
            response = requests.get(url)
            pdb_id = pdb_id.lower()
            pdb_info = response.json()[pdb_id]
            polypeptide_count = 0
            valid = False
            min_length = float('inf')
            max_length = 0
            entries = []
            for molecule in pdb_info:
                if molecule['molecule_type'] == 'polypeptide(L)':
                    polypeptide_count += 1
                    entry_id = molecule['entity_id']
                    length = molecule['length']
                    if length <= 50:
                        valid = True
                    entries.append((entry_id, length))
                    min_length = min(min_length, length)
                    max_length = max(max_length, length)
            if polypeptide_count >= 2 and valid:
                print(
                f"PDB ID: {pdb_id}, Polypeptide Entry Count: {polypeptide_count}, Max Length: {max_length}, Min Length: {min_length}")
                f.write(f"{pdb_id}\t{polypeptide_count}\t{max_length}\t{min_length}\t")
                for entry in entries:
                    f.write(f"{entry[0]}:{entry[1]}, ")
                f.write('\n')
                f.flush()

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_pdb_info, pdb_id) for pdb_id in pdb_ids]

from pypdb import *
import gzip

DATASET_DIR = "dataset/train"

def get_id(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            yield line.split("/")[1].split(".")[0]

if __name__ == "__main__":
    id_itr = get_id("training.txt")
    c = 0
    for id in id_itr:
        print(id)
        pdb_file = get_pdb_file(id, filetype='pdb', compression=False)
        with gzip.open("%s/%s.pdb.gz" %(DATASET_DIR, id), "wt") as f:
            f.write(pdb_file)
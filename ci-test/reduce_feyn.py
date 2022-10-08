import glob
import os
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__)) + \
    '/data/feynman/**/*.tsv.gz'
files = glob.glob(path, recursive=True)
for file in files:
    print(file)
    df = pd.read_csv(file, compression='gzip').sample(n=2000, random_state=43)
    df.to_csv(file, compression='gzip', index=False)

from HROCH import PHCRegressor
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import argparse
import yaml

import os

target_noise = 0.0


def read_metadata(file: str):
    with open(file, 'r') as e:
        content = yaml.safe_load(e)
        description = content['description'].split('\n')
        eq = [ms for ms in description if '=' in ms][0].split('=')[-1]
        return eq
    return None


def test_sample(name, timeLimit, numThreads, stoppingCriteria, verbose):
    f = pd.read_csv(name, sep="\t", compression='gzip')
    # normalize feature names
    f.columns = f.columns.str.replace(r"[-.]", "_", regex=True)

    Y = pd.DataFrame(f, columns=['target'])
    X = f.drop(columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75,
                                                        test_size=0.25, random_state=42)

    y_train = y_train.to_numpy()

    if target_noise > 0:
        noise = np.random.normal(0,
                                 target_noise *
                                 np.sqrt(
                                     np.mean(np.square(y_train))),
                                 size=[len(y_train), 1])
        y_train = y_train + noise

    reg = PHCRegressor(numThreads, timeLimit, stoppingCriteria,
                       "f32", "math", verbose=verbose)
    reg.fit(X_train.to_numpy(), y_train)
    yp = reg.predict(X_test)

    r2 = r2_score(y_test, yp)
    rms = np.sqrt(mean_squared_error(y_test.to_numpy(), yp))
    # get model string
    model_str = str(reg.sexpr)
    model_str = str(sp.parse_expr(model_str))

    mapping = {'x'+str(i+1): k for i, k in enumerate(X.columns)}
    new_model = model_str
    for k, v in reversed(mapping.items()):
        new_model = new_model.replace(k, v)

    return [rms, r2, new_model]


def all_samples(path: str, dataset: str, timeLimit: float, numThreads: int, stopingCriteria: float, verbose: bool = False):

    out_dir = os.path.join(path, 'results')
    data_dir = os.path.join(path, 'data/' + dataset)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = pd.DataFrame(
        columns=['name', 'r2', 'rms', 'true equation', 'found equation'])
    idx = 0
    fp = os.path.dirname(data_dir)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            suffix = '.tsv.gz'
            if file.endswith(suffix):
                name = file[:-len(suffix)]
                true_eq = read_metadata(os.path.join(root, 'metadata.yaml'))

                full_path = os.path.join(root, file)
                rms, r2, eq = test_sample(full_path, timeLimit, numThreads,
                                          stopingCriteria, verbose)
                results.loc[idx] = [name, r2, rms, true_eq, str(eq)]
                idx = idx + 1
                print(
                    f"{file}: r2={r2}, rms={rms} eq={str(eq)}")

    results.sort_values(by=['name'], inplace=True)
    csv_path = out_dir + '/' + dataset + '.csv'
    print('save to ', csv_path)
    results.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', default="", type=str)

    args = parser.parse_args()
    path = os.path.dirname(os.path.abspath(__file__))

    all_samples(path, args.dataset, numThreads=1, timeLimit=5.0,
                stopingCriteria=0.0, verbose=False)


if __name__ == "__main__":
    main()

import argparse
import pickle

def configPath():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", '-C',
        type=str, default="Experiments/config_default/",
        help="specific path to the configuration.py"
    )

    return parser.parse_args()

def save_dict(obj, fname):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)
import argparse
import os
import pandas as pd
from utils import read_config_file
from models.tfidf import TFIDF
from models.fasttext import FASTTEXT

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('type', type=str, help="The type of architecture to use.", choices=['tfidf','fasttext', 'bert'])
parser.add_argument('model_file', type=str,
                    help='Path to dataset csv file. (text file if of type khpos or directory if of type phylypo)')
parser.add_argument('source', type=str,
                    help='Path to dataset. (text file if of type khpos or directory if of type phylypo)', default="./datasets")
parser.add_argument('target', type=str,
                    help='Path to output directory.', default=".")
args = parser.parse_args()

config = read_config_file(args.config)

model = None
if args.type == "tfidf":
    model = TFIDF(config)
elif args.type == "fasttext":
    model = FASTTEXT(config)

model.load(args.model_file)

similarity = model.compare(args.source, args.target)

print(similarity)
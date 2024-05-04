import argparse
import os
from utils import read_config_file
from models.tfidf import TFIDF

parser = argparse.ArgumentParser(description='Run the training loop.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('type', type=str, help="The type of architecture to use.", choices=['tfidf','fasttext', 'bert'])
parser.add_argument('dataset_path', type=str,
                    help='Path to dataset. (text file if of type khpos or directory if of type phylypo)')
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.', default=".")
args = parser.parse_args()

print(args)


config = read_config_file(args.config)

pretrained_weight_path = os.path.join("pretrained", config['pretrained_weights'])

print(pretrained_weight_path)

exit()

model = None
if args.type == "tfidf":
    model = TFIDF(config)
elif args.type == "fasttext":
    pass

model.train()


print(config)
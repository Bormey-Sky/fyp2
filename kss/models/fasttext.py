from gensim.utils import tokenize
from gensim import utils
import numpy as np
from scipy.spatial import distance
from gensim.models.fasttext import FastText
from gensim.test.utils import get_tmpfile
from typing import List

from utils import preprocess


class Iter:
    def __init__(self, items: List[str]) -> None:
        self.items = items
        
    def __iter__(self):
        for item in self.items:
            with open(item, "r") as sample_file:
                words = preprocess(sample_file.read())
                yield words

class FASTTEXT():
    def __init__(self, config) -> None:
        self.model = FastText(
            sg={
                "cbow": 0,
                "skipgram": 1,
            }[config["model"]],
            vector_size=config["vector_size"],
            window=config["window"],
            epochs=config["epochs"],
            alpha=config["alpha"],
            min_n=config["min_n"],
            max_n=config["max_n"]
        )
        self.decision_threshold = config['decision_threshold']
        
        
    def train(self, datasets: str):
        datasets = datasets[datasets['class'] == 1]['source']
        self.model.build_vocab(
            corpus_iterable=Iter(datasets.tolist())
        )
        
        total_examples = self.model.corpus_count
        
        self.model.train(
            Iter(datasets.tolist()),
            epochs=self.model.epochs,
            total_examples=total_examples
        )
        
    def save(self, output_path: str):
        self.model.save(output_path)
        
    def compare(self, source, target):
        source = self.__doc_to_vec(source)
        target = self.__doc_to_vec(target)
        dist = distance.cosine(source, target)
        return dist, int(dist < self.decision_threshold)
        
    def __doc_to_vec(self, doc):
        with open(doc, "r") as doc_file:
            words = preprocess(doc_file.read())
            words = [self.model.wv[w] for w in words]
            words = np.array(words)
            words = np.mean(words, axis=0)
            return words
        
    def load(self, model_path: str):
        self.model = FastText.load(model_path)
        
            
        
        
        
   
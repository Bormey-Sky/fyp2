import json
import numpy as np
from tqdm import tqdm
from typing import List
from gensim import utils
from utils import preprocess
import matplotlib.pyplot as plt
from gensim.utils import tokenize
from scipy.spatial import distance
from gensim.test.utils import get_tmpfile
from gensim.models.fasttext import FastText



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
        self.similarity_measure = config["similarity_measure"]
        self.log_file = config["log_file"]
        self.patience = config["patience"]
        
        
    def train(self, datasets: str):
        datasets = datasets[datasets['class'] == 1]['source']
        self.model.build_vocab(
            corpus_iterable=Iter(datasets.tolist())
        )
        total_examples = self.model.corpus_count
        
        # self.model.train(
        #     Iter(datasets.tolist()),
        #     epochs=self.model.epochs,
        #     total_examples=total_examples
        # )
        
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        for epoch in tqdm(range(1, self.model.epochs + 1) , desc = "epochs"):
            self.model.train(
            Iter(datasets.tolist()),
            epochs=1,
            total_examples=total_examples
        )
            current_loss = self.compute_loss(datasets.tolist())
            losses.append({'epoch': epoch, 'loss': current_loss})
            
            with open(self.log_file, 'w') as file:
                json.dump(losses, file)
            
            self.model.save("best_model.model")
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                self.model.save("best_model.model") 
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        self.plot_losses(losses)
        
    def compute_loss(self, datasets):
        # Implement a method to compute loss, here using a proxy measure like average similarity
        total_loss = 0
        count = 0
        for doc in datasets:
            words = preprocess(open(doc).read())
            word_vectors = [self.model.wv[w] for w in words if w in self.model.wv]
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
                for word in word_vectors:
                    total_loss += distance.cosine(doc_vector, word)
                    count += 1
        return total_loss / count if count != 0 else 0
            
        
    def save(self, output_path: str):
        self.model.save(output_path)
        
    def compare(self, source, target):
        source = self.__doc_to_vec(source)
        target = self.__doc_to_vec(target)
        if self.similarity_measure == "cosine_similarity":
            dist = distance.cosine(source, target)
        elif self.similarity_measure == "euclidean_distance":
            dist = distance.euclidean(source, target)
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
        
    
    ## trying my best to plot ?
    def plot_losses(self, losses):
        epochs = [entry['epoch'] for entry in losses]
        loss_values = [entry['loss'] for entry in losses]
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss_values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)
        plt.show()
        
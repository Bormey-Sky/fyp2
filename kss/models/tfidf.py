class TFIDF:
    def __init__(self, config) -> None:
        self.similarity_measure = config["similarity_measure"]
    
    def train(self):
        raise NotImplementedError()
    
    def compare(self, source, target):
        raise NotImplementedError()
    
    def save(self, output_path: str):
        raise NotImplementedError()
    
    def calculate_similarity(self):
        if self.similarity_measure == "asdf":
            raise NotImplementedError()
    
if __name__ == "__main__":
    model = TFIDF()
    model.compare(None, None)
    model.train()
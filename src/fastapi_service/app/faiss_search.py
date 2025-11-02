import numpy as np
import faiss
from pathlib import Path

class FaissSearcher:
    def __init__(self, index_path='/models/faiss/index.faiss', names_path='/models/faiss/image_names.npy'):
        self.index_path = Path(index_path)
        self.names_path = Path(names_path)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None
        if self.names_path.exists():
            self.names = np.load(str(self.names_path), allow_pickle=True)
        else:
            self.names = []

    def search(self, embedding, top_k=5):
        if self.index is None:
            return []
        vec = embedding.astype('float32')
        faiss.normalize_L2(vec)
        dists, inds = self.index.search(vec, top_k+1)
        results = []
        for dist, idx in zip(dists[0], inds[0]):
            if idx>=0 and idx < len(self.names):
                results.append((self.names[idx], float(dist)))
        seen=set(); out=[]
        for n,s in results:
            if n not in seen:
                out.append((n,s)); seen.add(n)
            if len(out)>=top_k: break
        return out

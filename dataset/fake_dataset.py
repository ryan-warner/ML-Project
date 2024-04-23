import datasets
from huggingface_hub import hf_hub_download
import os
#from src import logger

class FakeDataset():
    current_tarball = None
    current_url = None
    hf_url = None
    dataset = None
    iterator = None
    train = True

    def __init__(self, hf_url, streaming=True, cache_dir="dataset"):
        self.current_tarball = None
        self.hf_url = hf_url
        logger.info( "Loading dataset")
        self.dataset = datasets.load_dataset(hf_url, streaming=streaming, cache_dir=cache_dir)
        logger.info("Dataset loaded")

    # Override the __iter__ method to allow for iteration over the dataset
    def __iter__(self):
        if self.train:
            self.iterator = iter(self.dataset['train'])
        else:
            self.iterator =  iter(self.dataset['validation'])
        
        return self
    
    # override the __next__ method to allow for iteration over the dataset
    def __next__(self):
        result = next(self.iterator)
        
        """ if result['__url__'] != self.current_url:
            # Delete the old dataset from disk
            if self.current_tarball:
                os.remove(self.current_tarball)

            subfolder = "ImageData/" + result['__url__'].split("/ImageData/")[-1].split(result['__url__'].split("/")[-1])[0][:-1]

            self.current_tarball = hf_hub_download(self.hf_url, result['__url__'].split("/")[-1], local_dir="dataset", cache_dir='dataset', repo_type='dataset', subfolder=subfolder)
            self.current_url = result['__url__'] """

        im = result['png']
        return im
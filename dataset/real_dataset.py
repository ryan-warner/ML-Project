from PIL import Image
import requests
from io import BytesIO
from src import logger

class RealDataset():
    df = None
    iterator = None
    train = True

    def __init__(self, df):
        self.df = df

    # Override __iter__ method to allow for iteration over the dataset
    def __iter__(self):
        self.iterator = iter(self.df['url'])
        return self
    
    # Override __next__ method to allow for iteration over the dataset - returl PIL image from URL
    def __next__(self):
        result = next(self.iterator)
        response = None
        while response == None:
            try:
                response = requests.get(result, timeout=5)
                if response.status_code != 200:
                    response = None
                    result = next(self.iterator)
                else:
                    im = Image.open(BytesIO(response.content))

            except Exception as e:
                logger.warning(e)
                result = next(self.iterator)
                response = None
                pass

        return im
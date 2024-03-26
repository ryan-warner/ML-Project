from datasets import load_dataset

dataset = load_dataset("InfImagine/FakeImageDataset", streaming=True, data_dir="/scratch")

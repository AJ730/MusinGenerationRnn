from torch.utils.data import RandomSampler, DataLoader

from datasets.datasets import SequenceDataset


class MusicDataLoader:
    def __init__(self, dataset: SequenceDataset, batch_size=64, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loader(self):
        sampler = RandomSampler(self.dataset, replacement=False)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers= self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )


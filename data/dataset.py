# data/dataset.py

import torch
from torch.utils.data import Dataset

from utils.attention_reg import load_feature_bundle


class FeatureDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of dicts:
        {
          "features": path to .pt (tensor [T,D] or dict with features/cnn_logits),
          "labels": path to .pt tensor [T]
        }
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        features, cnn_logits = load_feature_bundle(item["features"])
        labels = torch.load(item["labels"], map_location="cpu")

        T = min(features.shape[0], labels.shape[0])
        features = features[:T]
        labels = labels[:T]
        if cnn_logits is not None:
            cnn_logits = cnn_logits[:T]

        return features, labels, cnn_logits


def collate_variable_length(batch):
    features, labels, cnn_logits_list = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in features])
    max_len = max(lengths).item()
    dim = features[0].shape[1]
    num_classes = None
    if cnn_logits_list[0] is not None:
        num_classes = cnn_logits_list[0].shape[1]

    padded_features = torch.zeros(len(batch), max_len, dim)
    padded_labels = torch.full((len(batch), max_len), -100)
    padded_cnn_logits = None
    if num_classes is not None:
        padded_cnn_logits = torch.zeros(len(batch), max_len, num_classes)

    padding_mask = torch.ones(len(batch), max_len).bool()

    for i, (x, y) in enumerate(zip(features, labels)):
        T = x.shape[0]
        padded_features[i, :T] = x
        padded_labels[i, :T] = y
        padding_mask[i, :T] = False
        if padded_cnn_logits is not None and cnn_logits_list[i] is not None:
            padded_cnn_logits[i, :T] = cnn_logits_list[i]

    return padded_features, padded_labels, padding_mask, padded_cnn_logits

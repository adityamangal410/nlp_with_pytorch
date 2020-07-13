import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

class TweetsClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, 
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): size of the hidden dimension
            num_classes (int): number of classes in the classification
            dropout_p (float): a dropout parameter
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided,
            padding_idx (int): an index representing a null position
        """
        
        super(TweetsClassifier, self).__init__()
        
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim = embedding_size,
                                   num_embeddings = num_embeddings,
                                   padding_idx = padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim = embedding_size,
                                   num_embeddings = num_embeddings,
                                   padding_idx = padding_idx,
                                   _weight = pretrained_embeddings)
            
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                     out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                     out_channels=num_channels,
                     kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                     out_channels=num_channels,
                     kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels,
                     out_channels=num_channels,
                     kernel_size=3),
            nn.ELU()            
        )
        
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x_in, apply_softmax=True):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes).
        """
        
        x_embedded = self.emb(x_in).permute(0,2,1)
        
        features = self.convnet(x_embedded)
        
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p = self.dropout_p)
        
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self.dropout_p))
        prediction_vector = self.fc2(intermediate_vector)
        
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
            
        return prediction_vector
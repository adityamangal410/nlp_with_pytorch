import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

args = Namespace(
    raw_dataset_csv="../data/raw/nlp_with_disaster_tweets/train.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="../data/processed/nlp_with_disaster_tweets/train_with_splits.csv",
    seed=1337
)

# Read raw data
tweets = pd.read_csv(args.raw_dataset_csv, header=0)

# Splitting train by category
# Create dict
by_category = collections.defaultdict(list)
for _, row in tweets.iterrows():
    by_category[row.target].append(row.to_dict())
    
    
# Create split data
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_category.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_proportion*n)
    n_val = int(args.val_proportion*n)
    n_test = int(args.test_proportion*n)
    
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  
    
    # Add to final list
    final_list.extend(item_list)
    
# Write split data to file
final_tweets = pd.DataFrame(final_list)

# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
    
final_tweets.text = final_tweets.text.apply(preprocess_text)

final_tweets.head()

# Write munged data to CSV
final_tweets.to_csv(args.output_munged_csv, index=False)
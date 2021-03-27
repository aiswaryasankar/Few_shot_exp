from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import sys
from pkgutil import simplegeneric
from transformers import XLNetTokenizer, XLNetModel
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn

from config import DATA_PATH

# For SNIPS:

import unicodedata
import requests


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'val'):
            raise(ValueError, 'subset must be one of (train, val)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'val'):
            raise(ValueError, 'subset must be one of (train, val)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


class ClinicDataset(Dataset):

    def __init__(self, subset):

        if subset not in ('train', 'val', 'test'):
            raise(ValueError, 'subset must be one of (train, val, test)')

        self.subset = subset
        self.df = self.process_c150_data()
        self.df = self.df.assign(id=self.df.index.values)
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_len = 64

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.df["text"][item])
        label = self.df["label"][item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)

        return [input_ids, attention_mask.flatten(), torch.tensor(label, dtype=torch.long)]

    def process_c150_data(self):
        """
            Subset tells you whether to return train, val or test.
        """
        # Open a file: file
        fileName = os.getcwd() + '/data_small.json'
        file = open(fileName, mode='r')
        all_of_it = file.read()
        label_mapping = {}

        @simplegeneric
        def get_items(obj):
            while False: # no items, a scalar object
                yield None

        @get_items.register(dict)
        def _(obj):
            return obj.items() # json object. Edit: iteritems() was removed in Python 3

        @get_items.register(list)
        def _(obj):
            return enumerate(obj) # json array

        def strip_whitespace(json_data):
            for key, value in get_items(json_data):
                if hasattr(value, 'strip'): # json string
                    json_data[key] = value.strip()
                else:
                    strip_whitespace(value) # recursive call

        data = json.loads(all_of_it) # read json data from standard input
        strip_whitespace(data)

        labels = []
        for text, label in data[self.subset]:
            labels.append(label)

        label_set = set(labels)
        label_mapping = {}

        index = 0
        for label in label_set:
            label_mapping[label] = index
            index += 1

        # Convert into dataframe
        embedded = []

        for text, label in data[self.subset]:
            row = {"text": text, "label": label_mapping[label]}
            embedded.append(row)

        df = pd.DataFrame(embedded)

        return df


class SNIPSDataset(Dataset):

    def __init__(self):

        self.df = self.process_SNIPS_data()
        self.df = self.df.assign(id=self.df.index.values)
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.max_len = 64

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.df["text"][item])
        label = self.df["label"][item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)

        return [input_ids, attention_mask.flatten(), torch.tensor(label, dtype=torch.long)]

    def process_SNIPS_data(self):

        intents = ['AddToPlaylist','BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
        ds_types = ['train', 'validate']
        ds_suffixes = ['', '_full']
        url_list = []

        for intent in intents:
          for ds_type in ds_types:
              if ds_type == 'train':
                  for ds_suffix in ds_suffixes:
                      url = 'https://raw.githubusercontent.com/sonos/nlu-benchmark/master/2017-06-custom-intent-engines/' + \
                          intent + '/' + ds_type + '_' + intent + ds_suffix + '.json'
                      url_list.append(url)
              else:
                  url = 'https://raw.githubusercontent.com/sonos/nlu-benchmark/master/2017-06-custom-intent-engines/' + \
                      intent + '/' + ds_type + '_' + intent + '.json'   
                  url_list.append(url)

        for url in url_list:
          write_file = url.rsplit('/', 1)[-1]
          r = requests.get(url, allow_redirects=True)
          if r.status_code == 200:
              print('writing file ', write_file, '....')
              open(write_file, 'wb').write(r.content)
          else:
              print(write_file, ' Not Valid URL')

        

        def strip_accents(text):
          """
          Strip accents from input String.

          :param text: The input string.
          :type text: String.

          :returns: The processed String.
          :rtype: String.
          """
          try:
              text = unicode(text, 'utf-8')
          except (TypeError, NameError): # unicode is a default on python 3 
              pass
          text = unicodedata.normalize('NFD', text)
          text = text.encode('ascii', 'ignore')
          text = text.decode("utf-8")
          return str(text)

        ## Function to conver JSON to DF

        def get_df_from_json_file(intent_list, ds_type, label_map):
          
          text_arr = []

          for intent in intent_list:
              json_file = ds_type+ '_' + intent +'.json'
              with open(json_file, encoding='utf-8') as fh:
                  json_data = json.load(fh)
                  for i, example in enumerate(json_data[intent]):
                      utt_dict = {}
                      utt_arr = []
                      for dict_list in example['data']:
                          if 'text' in dict_list.keys():
                              text_accent_fix = strip_accents(dict_list['text']).lower()
                              utt_arr.append(text_accent_fix)

                      utter = ''.join(utt_arr)
                      utt_dict['text'] = utter
                      utt_dict['label'] = label_map[intent]

                      text_arr.append(utt_dict)
                              
          return pd.DataFrame(text_arr).sample(frac=1).reset_index(drop=True)    

        labels = ['AddToPlaylist','BookRestaurant',  'PlayMusic', 'SearchCreativeWork', 'SearchScreeningEvent', 'GetWeather', 'RateBook']

        label_map = {'AddToPlaylist' : 1,
                  'BookRestaurant' : 2,  
                  'PlayMusic' : 3, 
                  'SearchCreativeWork' : 4, 
                  'SearchScreeningEvent' : 5,
                  'GetWeather' : 6, 
                  'RateBook' : 7}

        train_df = get_df_from_json_file(intent_list=labels, ds_type='train', label_map = label_map)
        val_df = get_df_from_json_file(intent_list=labels, ds_type='validate', label_map = label_map)

        print('train_df: ', len(train_df))
        print('val_df: ', len(val_df))

        df = pd.concat([train_df, val_df] , ignore_index = True)

        return df


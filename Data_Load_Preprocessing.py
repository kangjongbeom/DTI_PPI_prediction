from tdc.multi_pred import DTI
from tdc.single_pred import HTS
from tdc import Evaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Data_Loader(dataset_name='KIBA', preprocessing=True): # 'DAVIS'
    if dataset_name == 'KIBA':
        data = DTI(name=dataset_name)
        print(data.print_stats())
        print('\nsplit data as train/valid/test....')
        split = data.get_split()
        train_df = split['train']
        valid_df = split['valid']
        test_df = split['test']
        if preprocessing:
            return data_preprocessing(train_df), data_preprocessing(valid_df), data_preprocessing(test_df)
        else:
            return train_df, valid_df, test_df
    elif dataset_name == 'DAVIS':
        data = DTI(name=dataset_name)
        print(data.print_stats())
        print('\nsplit data as train/valid/test....')
        split = data.get_split()

        train_df = split['train']
        valid_df = split['valid']
        test_df = split['test']
        print('return 3 dataframe {train_df, valid_df, test_df} !')
        if preprocessing:
            return data_preprocessing(train_df), data_preprocessing(valid_df), data_preprocessing(test_df)
        else:
            return train_df, valid_df, test_df
    else:
        print('Please check dataset_name, KIBA or DAIVS')

def data_preprocessing(df): # should delete raw datas have '.' in smiles datas
    print(f'Before preprocessing... Df Length is {len(df)}')
    salt_idx = []
    for num, i in enumerate(df.Drug):
        if '.' in list(i):
            salt_idx.append(num)
    salt_del_df = df.drop(salt_idx)
    print(f'After preprocessing... Df Length is {len(salt_del_df)} delete {len(df) - len(salt_del_df)}')
    return salt_del_df

def make_smi_vocab():
    data = HTS(name = 'HIV')
    smi_data = data.get_data().Drug # Only need smiles (for make vocab)
    smi_list = []
    for i in smi_data:
        smi_list += list(i)
    smi_list = sorted(set(smi_list))
    smi_vocab = {}
    for i,j in enumerate(smi_list):
        smi_vocab[j] = i+1
    smi_vocab['<PAD>'] = 0
    return smi_vocab

smi_vocab = {'#': 1, '%': 2, '(': 3, ')': 4, '+': 5, '-': 6, '.': 7,
             '0': 8, '1': 9, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14,
             '7': 15, '8': 16, '9': 17, '=': 18, 'A': 19, 'B': 20, 'C': 21,
             'F': 22, 'G': 23, 'H': 24, 'I': 25, 'K': 26, 'L': 27, 'M': 28,
             'N': 29, 'O': 30, 'P': 31, 'R': 32, 'S': 33, 'T': 34, 'U': 35,
             'V': 36, 'W': 37, 'Z': 38, '[': 39, ']': 40, 'a': 41, 'b': 42,
             'c': 43, 'd': 44, 'e': 45, 'g': 46, 'h': 47, 'i': 48, 'l': 49,
             'n': 50, 'o': 51, 'p': 52, 'r': 53, 's': 54, 't': 55, 'u': 56,
             '<PAD>': 0}


# amino_vocab is so easy just 20 amino_acid!!

amino_vocab = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
               'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
               'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
               'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
               '<PAD>': 0}

# amino_vocab['<PAD>'] = 0 # add '<PAD>' token


class dti_tokenize():
  def __init__(self, smi_vocab, target_vocab):
    self.drug_vocab = smi_vocab
    self.target_vocab = target_vocab
    self.drug_Length = 100
    self.target_Length = 1000

  def padding(self, encoded, drug=True):
    if drug:
      return encoded + (self.drug_Length - len(encoded)) * [self.drug_vocab['<PAD>']]
    else :
      return encoded + (self.target_Length - len(encoded)) * [self.target_vocab['<PAD>']]

  def drug_encode(self, smiles):
    encoded = [self.drug_vocab[i] for i in smiles[:self.drug_Length]]
    return np.array(self.padding(encoded, True))

  def target_encode(self, target):
    encoded = [self.target_vocab[i] for i in target[:self.target_Length]]
    return np.array(self.padding(encoded, False))


class make_model_data():
    def __init__(self,dataset_name ='KIBA'):
        self.data_Loader = Data_Loader(dataset_name,True)
        self.drug_vocab = smi_vocab
        self.target_vocab = amino_vocab
        self.tokenize = dti_tokenize(self.drug_vocab, self.target_vocab)

    def load_data(self): # preprocessing 포함
        train, valid, test = self.data_Loader
        return train, valid, test

    def split_feature_label(self,df):
        drug_token = [self.tokenize.drug_encode(i) for i in df.Drug.copy()]
        target_token = [self.tokenize.target_encode(i) for i in df.Target.copy()]
        label = df.Y.copy()
        return drug_token, target_token, label

    def model_data(self, mode='train'):
        train, valid, test = self.load_data()
        if mode == 'train':
            train_drug, train_target, train_label = self.split_feature_label(train)
            return np.vstack(train_drug), np.vstack(train_target), train_label
        elif mode == 'valid':
            valid_drug, valid_target, valid_label = self.split_feature_label(valid)
            return np.vstack(valid_drug), np.vstack(valid_target), valid_label
        elif mode == 'test':
            test_drug, test_target, test_label = self.split_feature_label(test)
            return np.vstack(test_drug), np.vstack(test_target), test_label
        else:
            print('Check mode {train, valid, test}')
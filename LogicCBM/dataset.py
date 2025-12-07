"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import _pickle as pkl
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import *
import pandas as pd
from PIL import Image, ImageFilter
import random
from collections import defaultdict

import sys
sys.path.append('<PROJECT_PATH>')

from LogicCBM.config import BASE_DIR
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader, Subset
import json
from collections import OrderedDict

DATA_DIR = './data'   

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sampleuse_attr
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples

def load_data(dataset, pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, split='train', n_class_attr=2, image_dir='images', resampling=False, resol=299, apply_corruption=False, ordering=None):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    if dataset == "cub":
        is_training = (split=="train")
        if is_training:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                ])
            
        dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, split, transform, apply_corruption=apply_corruption)

        if is_training:
            drop_last = True
            shuffle = True
        else:
            drop_last = False
            shuffle = False
        if resampling:
            sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
            loader = DataLoader(dataset, batch_sampler=sampler)
        elif ordering:
            ordered_subset = Subset(dataset, ordering)
            loader = DataLoader(ordered_subset, batch_size=batch_size, shuffle=False)  # shuffle=False preserves order
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return loader
    elif dataset == "cifar100":
        is_training = (split=="train")
        dataset = Cifar100Loader(split=split, apply_corruption=apply_corruption)
        if is_training:
            drop_last = True
            shuffle = True
        else:
            drop_last = False
            shuffle = False
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle, drop_last=drop_last)
        return dataloader    
    elif dataset == "awa2":
        return load_awa2_data(data_dir=DATA_DIR + '/data/AWA2/Animals_with_Attributes2/', batch_size=batch_size, split=split, apply_corruption=apply_corruption)

def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, split="train", transform=None, apply_corruption=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = (split=="train")
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.pathtoid = defaultdict(int)
        
        for i in range(len(self.data)):
            id, img_path, _, _, _, _ = self.data[i]['id'], self.data[i]['img_path'], self.data[i]['class_label'], self.data[i]['attribute_label'], self.data[i]['uncertain_attribute_label'], self.data[i]['attribute_label']
            self.pathtoid[img_path] = id

    @classmethod
    def select_concept_pairs(cls, instance, num_pairs):
        # Randomly select 30 samples from different classes and do a dot product between their concept vectors
        num_samples = range(instance.__len__())
        concept_pairs = []
        for i in range(num_pairs):
            num_inds = np.random.choice(range(12))
            inds = np.random.choice(num_samples, int(num_inds), replace=False)
            samples = [instance.__getitem__(i) for i in inds]
            cumulative_presence = torch.zeros(312)
            for sample in samples:
                cumulative_presence += sample[2]
            _, top_indices = torch.topk(cumulative_presence, 2)
            concept_pairs.append(top_indices.tolist())
        return torch.tensor(concept_pairs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']

        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, torch.tensor(attr_label)
        else:
            return img, class_label
    

class Cifar100Loader(Dataset):
    def __init__(
            self,
            json_file=DATA_DIR + 'data/cifar100/cifar100_filtered_new.json', 
            concept_label_dict = {},
            split='train',
            seed=None, 
            transforms=None,
        ):
        self.class_concept_dict = json.load(open(json_file, 'r'), object_pairs_hook=OrderedDict)
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.class_label_map = {i: k for i, k in zip(np.arange(0, 100), self.class_list)}
        self.concept_label_dict = concept_label_dict
        self.label_concept_dict = {}
        if transforms is None:
            if split == 'train':
                self.transforms = Compose([
                    Resize((308, 308)), 
                    RandomCrop(299),    
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = Compose([
                    Resize((308, 308)), 
                    CenterCrop(299),    
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

        self.data = torchvision.datasets.CIFAR100(root=DATA_DIR + '/data/cifar100', train=(split=='train'), download=False, transform=self.transforms)
        class_list = list(range(len(self.class_concept_dict.keys())))

        self.concept_vectors = []
        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            attr_values = []
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys()) 
                    self.label_concept_dict[len(self.concept_label_dict.keys())] = attr
                    attr_values.append(self.concept_label_dict[attr])
            concept_vector = np.zeros(self.get_concept_count())
            concept_vector[attr_values] = 1
            self.concept_vectors.append(concept_vector)

    @classmethod
    def select_concept_pairs(cls, instance, num_pairs):
        # Randomly select 30 samples from different classes and do a dot product between their concept vectors
        num_samples = range(instance.__len__())
        concept_pairs = []
        for i in range(num_pairs):
            num_inds = np.random.choice(range(12))
            inds = np.random.choice(num_samples, int(num_inds), replace=False)
            samples = [instance.__getitem__(i) for i in inds]
            cumulative_presence = torch.zeros(925)
            for sample in samples:
                cumulative_presence += sample[2]
            _, top_indices = torch.topk(cumulative_presence, 2)
            concept_pairs.append(top_indices.tolist())
        return torch.tensor(concept_pairs)

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        image, label = self.data[index]
        attrs = self.concept_vectors[label]

        return image, label, attrs
     
class AnimalDataset(Dataset):
  def __init__(self, data_dir, classes_file, transform, split='train'):
    predicate_binary_mat = np.array(np.genfromtxt(data_dir + '/predicate-matrix-binary.txt', dtype='int'))
    self.predicate_binary_mat = predicate_binary_mat
    self.transform = transform
    self.split = 'train' if split == 'train' else 'test'

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(data_dir + '/classes.txt') as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    df = pd.read_csv(data_dir + '/{}_logic.csv'.format(split))
    self.img_names = df['img_name'].tolist()
    self.img_index = df['img_index'].tolist()

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)
    if self.apply_corruption:
        corruption_type = 'noise'
        im = self.corruption(im, corruption_type)

    im_index = self.img_index[index]
    im_predicate = self.predicate_binary_mat[im_index,:]
    return im, im_index, im_predicate       # image, class_label, attribute_label

  def __len__(self):
    return len(self.img_names)      

  @classmethod
  def select_concept_pairs(cls, instance, num_pairs):
    # Randomly select 30 samples from different classes and do a dot product between their concept vectors
    num_samples = range(instance.__len__())
    concept_pairs = []
    for i in range(num_pairs):
        num_inds = np.random.choice(range(12))
        inds = np.random.choice(num_samples, int(num_inds), replace=False)
        samples = [instance.__getitem__(i) for i in inds]
        cumulative_presence = torch.zeros(85)
        for sample in samples:
            cumulative_presence += sample[2]
        _, top_indices = torch.topk(cumulative_presence, 2)
        concept_pairs.append(top_indices.tolist())
    return torch.tensor(concept_pairs)
  
def load_awa2_data(data_dir=DATA_DIR + '/data/AWA2/Animals_with_Attributes2/', batch_size=32, split='train'):
    if split == 'train':
        transform = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if split == 'train':
        classes_file = 'trainclasses.txt'
        drop_last = True
        shuffle = True
    else:
        classes_file = 'testclasses.txt'
        drop_last = False
        shuffle = False

    dataset = AnimalDataset(data_dir, classes_file, transform, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=8)
    return dataloader
import io
import os.path
from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
import tensorflow as tf


DATA_PATH = "../data/tfrecords-jpeg-512x512"


class FlowerDataset(Dataset):
    def __init__(self, images, classes, ids, transform, mode):
        self.images = images
        self.classes = classes
        self.ids = ids
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]))
        img = self.transform(img)
        if self.mode == "test":
            idd = self.ids[idx]
            return (img, idd)
        else:
            label = self.classes[idx]
            return (img, label)
        

feats = {
    'train': {
        'class': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    },
    'val': {
        'class': tf.io.FixedLenFeature([], tf.int64),
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    },
    'test': {
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
}

data_elems = {
    'train': {'ids': [], 'images': [], 'classes': []},
    'val': {'ids': [], 'images': [], 'classes': []},
    'test': {'ids': [], 'images': []}
}

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # https://pytorch.org/hub/pytorch_vision_resnet/
])

for dset_name in ['train', 'val', 'test']:
    dataset_path = os.path.join(DATA_PATH, dset_name)
    data_list = map(lambda path: tf.data.TFRecordDataset(os.path.join(dataset_path, path)), os.listdir(dataset_path))
    for tl in data_list:
        for t in tl.map(lambda x: tf.io.parse_single_example(x, feats[dset_name])):
            data_elems[dset_name]['ids'].append(str(t["id"].numpy())[2:-1])
            data_elems[dset_name]['images'].append(t["image"].numpy())
            if dset_name != 'test':
                data_elems[dset_name]['classes'].append(t["class"].numpy())

train_ds = FlowerDataset(data_elems['train']['images'], data_elems['train']['classes'], None, transform, 'train')
val_ds = FlowerDataset(data_elems['val']['images'], data_elems['val']['classes'], None, transform, 'val')
test_ds = FlowerDataset(data_elems['test']['images'], None, data_elems['test']['ids'], transform, 'test')

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)
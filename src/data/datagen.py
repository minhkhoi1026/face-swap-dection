import numpy as np
import os
import keras
import cv2
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import one_hot

from src.utils.retinex import automatedMSRCR

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, num_classes, batch_size=32, dim=(32, 32), type_gen='train'):
        '''Initialization'''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.num_classes = num_classes
        self.list_IDs = list_IDs
        self.type_gen = type_gen
        self.aug_gen = ImageDataGenerator()
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def sequence_augment(self, img):
        dictkey_list = ['theta', 'ty', 'tx',
                        'brightness', 'flip_horizontal', 'zy',
                        'zx']
        random_aug = np.random.randint(2, 5)  # random 2-4 augmentation method
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False)  #

        dict_input = {}
        for i in pick_idx:
            if dictkey_list[i] == 'theta': # rotate
                dict_input['theta'] = np.random.randint(-10, 10)

            elif dictkey_list[i] == 'ty':  # width_shift
                dict_input['ty'] = np.random.randint(-20, 20)

            elif dictkey_list[i] == 'tx':  # height_shift
                dict_input['tx'] = np.random.randint(-20, 20)

            elif dictkey_list[i] == 'brightness':
                dict_input['brightness'] = np.random.uniform(0.75, 1.25)

            elif dictkey_list[i] == 'flip_horizontal':
                dict_input['flip_horizontal'] = bool(random.getrandbits(1))

            elif dictkey_list[i] == 'zy':  # width_zoom
                dict_input['zy'] = np.random.uniform(0.75, 1.25)

            elif dictkey_list[i] == 'zx':  # height_zoom
                dict_input['zx'] = np.random.uniform(0.75, 1.25)
        img = self.aug_gen.apply_transform(img, dict_input)
        return img

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, self.dim[0], self.dim[1], 3)), 
             np.empty((self.batch_size, self.dim[0], self.dim[1], 3))]
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            img = cv2.imread(ID)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.dim[1], self.dim[0]))

            if self.type_gen =='train':
                img = self.sequence_augment(img)
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            new_img = np.expand_dims(new_img, -1)
            new_img = automatedMSRCR(new_img, [10, 20, 30])
            new_img = cv2.cvtColor(new_img[:, :, 0], cv2.COLOR_GRAY2RGB)

            X[0][i] = img/255.0
            X[1][i] = new_img/255.0

            y[i] = self.labels[ID]

        if self.num_classes > 1:
            y = one_hot(y, depth=self.num_classes)
            
        return X, y

def load_dataset_to_generator(data_path, num_classes, bs, dim, type_gen, oversampling=False):
    image_paths = load_image_file_paths(data_path, oversampling)
    labels = generate_label_from_path(image_paths)
    return DataGenerator(image_paths, labels, num_classes, batch_size=bs, dim=dim, type_gen=type_gen)

def load_image_file_paths(data_path, oversampling=False):
    image_paths = []
    
    dirs = [os.path.join(data_path, "fake"), os.path.join(data_path, "real")]
    if oversampling == True:
        nsample = max(len(os.listdir(x)) for x in dirs) 

        for folder in dirs:
            tmp_paths = []
            for path in os.listdir(folder):
                tmp_paths.append(os.path.join(folder, path))
            ids = np.arange(len(tmp_paths))
            choices = np.random.choice(ids, nsample)
            image_paths.extend([tmp_paths[id] for id in choices])
            print(len(image_paths))
    else:
          for folder in dirs:
            for path in os.listdir(folder):
                image_paths.append(os.path.join(folder, path))

    return image_paths

def generate_label_from_path(image_paths):
    labels = {}
    for path in image_paths:
        labels[path] = int(os.path.basename(os.path.dirname(path)) == 'real')
    return labels

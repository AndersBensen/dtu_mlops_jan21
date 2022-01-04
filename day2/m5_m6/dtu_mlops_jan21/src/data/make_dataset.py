# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


def save_tensor(tensor, path):
    torch.save(tensor, path)



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    test_ids = []
    test_path = input_filepath + "/test.npz"
    test = np.load(test_path)
    test_images = [i for i in test['images']]
    test_labels = [i for i in test['labels']]

    for i in range(len(test_labels)):
        test_ids.append((torch.tensor(test_images[i]), torch.tensor(test_labels[i])))
    
    torch.save(test_ids, output_filepath+"/test_tensor.pt")

    train_paths = [input_filepath + "/train_" + str(i)+".npz" for i in range(5)]    
    train_ids = []
    trains = [np.load(i) for i in train_paths]
    train_images = [i['images'] for i in trains]
    train_labels = [i['labels'] for i in trains]

    images_concat = np.concatenate(train_images, axis=0)
    labels_concat = np.concatenate(train_labels, axis=0)
    for i in range(len(labels_concat)):
        train_ids.append((torch.tensor(images_concat[i]),torch.tensor(labels_concat[i])))
    
    torch.save(train_ids, output_filepath+"/train_tensor.pt")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

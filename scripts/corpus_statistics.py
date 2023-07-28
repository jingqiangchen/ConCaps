import logging
import os
import random
import re
from typing import Dict

import numpy as np
import pymongo 
import torch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader 
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import Compose, Normalize, ToTensor 
from tqdm import tqdm

from tell.data.fields import ImageField, ListTextField, IndexField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def get_named_entities(section):
    # These name indices have the right end point excluded
    names = set()

    if 'named_entities' in section:
        ners = section['named_entities']
        for ner in ners:
            if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                names.add(ner['text'])

    return names

def get_person_names(section):
    # These name indices have the right end point excluded
    names = set()

    if 'named_entities' in section:
        ners = section['named_entities']
        for ner in ners:
            if ner['label'] in ['PERSON']:
                names.add(ner['text'])

    return names

def compute_statistics(split: str):
    image_dir = "/home/test/images_processed_dailymail"
    client = MongoClient(host='localhost', port=27017)
    db = client.dailymail
    roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
    if split not in ['train', 'valid', 'test']:
        raise ValueError(f'Unknown split: {split}')

    logger.info('Grabbing all article IDs')
    sample_cursor = db.articles.find({
        'split': split, 'n_images': {"$lte": 10, "$gte": 1}
    }, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    if split == "valid":
        ids = ids[:len(ids)//6]
    elif split == "test":
        ids = ids[:len(ids)//2]
    sample_cursor.close()
    rs = np.random.RandomState(1234)
    rs.shuffle(ids)

    projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                  'parsed_section.hash', 'parsed_section.parts_of_speech',
                  'parsed_section.facenet_details', 'parsed_section.named_entities',
                  'image_positions', 'headline',
                  'web_url', 'n_images_with_faces']

    for article_index, article_id in enumerate(ids):
        article = db.articles.find_one(
            {'_id': {'$eq': article_id}}, projection=projection)
        sections = article['parsed_section']
        image_positions = article['image_positions']
        image_index = -1
        image_count = 0
            
        for pos in image_positions:
            caption = sections[pos]['text'].strip()
            if not caption:
                continue
            image_path = os.path.join(
                image_dir, f"{sections[pos]['hash']}.jpg")
            if not os.path.exists(image_path):
                continue
            image_count += 1
            
        for pos in image_positions:
            title = ''
            if 'main' in article['headline']:
                title = article['headline']['main'].strip()
            paragraphs = []
            named_entities = set()
            n_words = 0
            if title:
                paragraphs.append(title)
                named_entities.union(
                    get_named_entities(article['headline']))
                n_words += len(to_token_ids(title))

            caption = sections[pos]['text'].strip()
            if not caption:
                continue

            before = []
            after = []
            i = pos - 1
            j = pos + 1
            for k, section in enumerate(sections):
                if section['type'] == 'paragraph':
                    paragraphs.append(section['text'])
                    named_entities |= get_named_entities(section)
                    break

            while True:
                if i > k and sections[i]['type'] == 'paragraph':
                    text = sections[i]['text']
                    before.insert(0, text)
                    named_entities |= get_named_entities(sections[i])
                    n_words += len(to_token_ids(text))
                i -= 1

                if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                    text = sections[j]['text']
                    after.append(text)
                    named_entities |= get_named_entities(sections[j])
                    n_words += len(to_token_ids(text))
                j += 1

                if n_words >= 510 or (i <= k and j >= len(sections)):
                    break
                
            image_path = os.path.join(
                image_dir, f"{sections[pos]['hash']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            paragraphs = paragraphs + before + after
            named_entities = sorted(named_entities)

            #image_id = article_index * 10000 + image_no
            image_index += 1








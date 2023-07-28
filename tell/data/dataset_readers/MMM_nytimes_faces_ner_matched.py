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


@DatasetReader.register('mmm_nytimes_faces_ner_matched')
class MMMNYTimesFacesNERMatchedReader(DatasetReader):
    """Read from the New York Times dataset.

    See the repo README for more instruction on how to download the dataset.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 use_caption_names: bool = True,
                 use_objects: bool = False,
                 n_faces: int = None,
                 n_fakes: int = 5,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.nytimes
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        
        self.n_fakes = n_fakes 
        
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({
            'split': split,
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        self.rs.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']

        for article_index, article_id in enumerate(ids):
            article = self.db.articles.find_one(
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
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
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
                        self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

                caption = sections[pos]['text'].strip()
                if not caption:
                    continue
                
                fake_captions = []
                n_fakes = 0
                while n_fakes < self.n_fakes:
                    fake_index = random.randint(0, len(ids) - 1)
                    fake_article_id = ids[fake_index]
                    if fake_article_id == article_id:
                        continue
                    fake_article = self.db.articles.find_one(
                        {'_id': {'$eq': fake_article_id}}, projection=projection)
                    fake_sections = fake_article['parsed_section']
                    fake_image_positions = fake_article['image_positions']
                    for fake_pos in fake_image_positions:
                        fake_caption = fake_sections[fake_pos]['text'].strip()
                        if fake_caption:
                            fake_captions.append(fake_caption)
                            n_fakes += 1 
                            break

                if self.n_faces is not None:
                    n_persons = self.n_faces
                elif self.use_caption_names:
                    n_persons = len(self._get_person_names(sections[pos]))
                else:
                    n_persons = 4

                before = []
                after = []
                i = pos - 1
                j = pos + 1
                for k, section in enumerate(sections):
                    if section['type'] == 'paragraph':
                        paragraphs.append(section['text'])
                        named_entities |= self._get_named_entities(section)
                        break

                while True:
                    if i > k and sections[i]['type'] == 'paragraph':
                        text = sections[i]['text']
                        before.insert(0, text)
                        named_entities |= self._get_named_entities(sections[i])
                        n_words += len(self.to_token_ids(text))
                    i -= 1

                    if k < j < len(sections) and sections[j]['type'] == 'paragraph':
                        text = sections[j]['text']
                        after.append(text)
                        named_entities |= self._get_named_entities(sections[j])
                        n_words += len(self.to_token_ids(text))
                    j += 1

                    if n_words >= 510 or (i <= k and j >= len(sections)):
                        break
                
                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue

                if 'facenet_details' not in sections[pos] or n_persons == 0:
                    face_embeds = np.array([[]])
                else:
                    face_embeds = sections[pos]['facenet_details']['embeddings']
                    # Keep only the top faces (sorted by size)
                    face_embeds = np.array(face_embeds[:n_persons])

                paragraphs = paragraphs + before + after
                named_entities = sorted(named_entities)

                obj_feats = None
                if self.use_objects:
                    obj = self.db.objects.find_one(
                        {'_id': sections[pos]['hash']})
                    if obj is not None:
                        obj_feats = obj['object_features']
                        if len(obj_feats) == 0:
                            obj_feats = np.array([[]])
                        else:
                            obj_feats = np.array(obj_feats)
                    else:
                        obj_feats = np.array([[]])
                        
                #image_id = article_index * 10000 + image_no
                image_index += 1

                yield self.article_to_instance(
                    image_count, article_index, image_index, paragraphs, named_entities, image, caption, fake_captions, image_path,
                    article['web_url'], pos, face_embeds, obj_feats)

    def article_to_instance(self, image_count, article_index, image_index, paragraphs, named_entities, image, caption, fake_captions, 
                            image_path, web_url, pos, face_embeds, obj_feats) -> Instance:
        context = '\n'.join(paragraphs).strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)
        
        fake_captions_tokens = [self._tokenizer.tokenize(caption) for caption in fake_captions]
        
        name_token_list = [self._tokenizer.tokenize(n) for n in named_entities]

        if name_token_list:
            name_field = [TextField(tokens, self._token_indexers)
                          for tokens in name_token_list]
        else:
            stub_field = ListTextField(
                [TextField(caption_tokens, self._token_indexers)])
            name_field = stub_field.empty_field()

        image_filed = ImageField(image, self.preprocess)
        
        fields = {
            "image_count": IndexField(image_count, image_filed),
            "article_index": IndexField(article_index, image_filed),
            "image_index": IndexField(image_index, image_filed),
            'context': TextField(context_tokens, self._token_indexers),
            'names': ListTextField(name_field),
            'image': image_filed,
            'caption': TextField(caption_tokens, self._token_indexers),
            'fake_captions': ListTextField([TextField(fake_caption_tokens, self._token_indexers) for fake_caption_tokens in fake_captions_tokens]),
            'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
        }

        if obj_feats is not None:
            fields['obj_embeds'] = ArrayField(obj_feats, padding_value=np.nan)

        metadata = {'context': context,
                    'caption': caption,
                    'names': named_entities,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def _get_named_entities(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids

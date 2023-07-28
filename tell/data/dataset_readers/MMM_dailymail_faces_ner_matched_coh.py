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

import json 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


@DatasetReader.register('MMM_dailymail_faces_ner_matched_coh')
class MMMDailyMailFacesNERMatchedReader2_coh(DatasetReader): 
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
                 fake_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 use_caption_names: bool = True,
                 use_objects: bool = False,
                 n_faces: int = None,
                 n_fakes: int = 5,
                 gen_coh: str = "coh",
                 gen_path: str = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_fc2/serialization/generations_greedy.jsonl",
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._fake_indexers = fake_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.dailymail
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        
        self.n_fakes = 1 
        self.gen_coh = gen_coh 
        self.gen_path = gen_path 
        
        self.use_entities = ["PERSON", "ORG", "GPE", "DATE"]
        
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices
        
        self.entities = {}
        person_cursor = self.db.entities.find({'label': "PERSON"})
        self.entities["PERSON"] = np.array([record['text'] for record in person_cursor])
        person_cursor.close()
        
        org_cursor = self.db.entities.find({'label': "ORG"})
        self.entities["ORG"] = np.array([record['text'] for record in org_cursor])
        org_cursor.close()
        
        gpe_cursor = self.db.entities.find({'label': "GPE"})
        self.entities["GPE"] = np.array([record['text'] for record in gpe_cursor])
        gpe_cursor.close()
        
        date_cursor = self.db.entities.find({'label': "DATE"})
        self.entities["DATE"] = np.array([record['text'] for record in date_cursor])
        date_cursor.close()
        
        self.gen_captions = {}
        if os.path.exists(self.gen_path): 
            with open(self.gen_path) as f:
                for line in f.readlines():
                    obj = json.loads(line)
                    image_id = os.path.splitext(os.path.basename(obj["image_path"]))[0] 
                    #print(image_id)
                    self.gen_captions[image_id] = obj["generation"] 

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        # validation and test sets contain 10K examples each
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({
            'split': split, 'n_images': {"$lte": 10, "$gte": 1}
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        if split == "valid":
            ids = ids[:len(ids)//6]
        elif split == "test":
            ids = ids[:len(ids)//2]
        sample_cursor.close()
        self.rs.shuffle(ids)
        
        ids = ["aa1fc22a78bdaebcfb711409eef190711fe8d299"]
        
        #ids = ['570df37438f0d804a21a402b', '5ba7b15900a1bc2872e89ea2', '5ba7b15900a1bc2872e89ea2', '5ba7b15900a1bc2872e89ea2', 
        #       '5ba7b15900a1bc2872e89ea2', '5ba7b15900a1bc2872e89ea2', '5ba7b15900a1bc2872e89ea2', '5438050a38f0d83c143b7f59', 
        #       '5334c48138f0d8100c38f07f', 
        #       '58a3f1377c459f2525d1c74d', '58a3f1377c459f2525d1c74d', '547c87de38f0d813efccc45c', '5487f84b38f0d8602128ec05', 
        #       '5487f84b38f0d8602128ec05', '5a2e10f395d0e0246f21b4d8']
        
        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'parsed_section.has_PGOD', 'image_positions', 'headline',
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
                
                if split == "test" and os.path.exists(self.gen_path):
                    hash = sections[pos]['hash']
                    if hash in self.gen_captions:
                        caption = self.gen_captions[hash] 
                    else:
                        continue
                
                fake_captions = []
                fake_caption = ""
                old_caption = sections[pos]['text']
                begin = 0
                #if "has_PGOD" not in sections[pos]:
                #    print(article_id)
                if sections[pos]["has_PGOD"]:
                    nes = sections[pos]["named_entities"]
                    for ne in nes:
                        if ne["label"] in self.use_entities:
                            while True:
                                ch = np.random.choice(self.entities[ne["label"]])
                                if ch != ne["text"]:
                                    break
                            fake_caption += old_caption[begin:ne["start"]]
                            fake_caption += ch
                            begin = ne["end"]
                    fake_caption += old_caption[begin:]
                    #print("--------------")
                    #print(article_id)
                    #print(old_caption)
                    #print(fake_caption)
                else:
                    has_noun = False
                    if "parts_of_speech" in sections[pos]:
                        words = sections[pos]["parts_of_speech"]
                        for word in words:
                            if word["pos"] in ["NOUN", "PROPN"]:
                                has_noun = True
                                while True:
                                    ch = np.random.choice(self.entities[np.random.choice(self.use_entities)])
                                    if ch != word["text"]:
                                        break
                                fake_caption += old_caption[begin:word["start"]]
                                fake_caption += ch
                                begin = word["end"]
                        fake_caption += old_caption[begin:]
                            
                    if not has_noun:
                        words = old_caption.split()
                        ch = np.random.choice(self.entities[np.random.choice(self.use_entities)])
                        words[np.random.randint(len(words))] = ch
                        fake_caption = " ".join(words)
                        
                if fake_caption.strip() == "":
                    continue
                
                
                
                fake_captions.append(fake_caption.strip())

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
                    "" if 'web_url' not in article else article['web_url'], pos, face_embeds, obj_feats, article_id) 

    def article_to_instance(self, image_count, article_index, image_index, paragraphs, named_entities, image, caption, fake_captions, 
                            image_path, web_url, pos, face_embeds, obj_feats, article_id) -> Instance:
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
            'fake_captions': ListTextField([TextField(fake_caption_tokens, self._fake_indexers) for fake_caption_tokens in fake_captions_tokens]),
            'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
        }

        if obj_feats is not None:
            fields['obj_embeds'] = ArrayField(obj_feats, padding_value=np.nan)

        metadata = {'context': context,
                    'caption': caption,
                    'names': named_entities,
                    'web_url': web_url,
                    'image_path': image_path,
                    'image_pos': pos,
                    "article_id": article_id}
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

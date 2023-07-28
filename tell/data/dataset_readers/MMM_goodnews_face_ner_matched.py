import logging
import os
import random
from typing import Dict

import numpy as np
import pymongo
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

import time

from tell.data.fields import ImageField, ListTextField, IndexField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('MMM_goodnews_face_ner_matched')
class MMMGoodNewsFaceNERMatchedReader2(DatasetReader):
    """Read from the Good News dataset.

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
                 eval_limit: int = 5120,
                 use_caption_names: bool = True,
                 use_objects: bool = False,
                 n_faces: int = None,
                 n_fakes: int = 5,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._fake_indexers = fake_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.goodnews
        self.image_dir = image_dir
        self.preprocess = Compose([
            # Resize(256), CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.eval_limit = eval_limit
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        
        self.n_fakes = 1 
        self.use_entities = ["PERSON", "ORG", "GPE", "DATE"]
        
        random.seed(1234)
        self.rs = np.random.RandomState(1234)
        
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

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        # Setting the batch size is needed to avoid cursor timing out
        # We limit the validation set to 1000
        limit = self.eval_limit if split == 'val' else 0

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.splits.find({
            'split': {'$eq': split},
        }, projection=['_id'], limit=limit).sort('_id', pymongo.ASCENDING)

        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        #self.rs.shuffle(ids)
        
        article_index = 0
        pre_article_id = ""
        image_count = 0
        for sample_id in ids:
            try:
                sample = self.db.splits.find_one({'_id': {'$eq': sample_id}})
    
                # Find the corresponding article
                article = self.db.articles.find_one({
                    '_id': {'$eq': sample['article_id']},
                }, projection=['_id', 'context', 'images', 'web_url', 'caption_ner', 'context_ner', "captions_has_PGOD"])
            except:
                os.system("mongod --bind_ip_all --dbpath /home/test/mongodb --wiredTigerCacheSizeGB 10 &")
                time.sleep(10)
                continue

            # Load the image
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue
            
            image_index = sample["image_index"]
            if sample["article_id"] != pre_article_id:
                article_index += 1
                pre_article_id = sample["article_id"]
                image_count = 0
            image_count += 1
            
            named_entities = sorted(self._get_named_entities(article))
            
            fake_captions = []
            fake_caption = ""
            old_caption = article["images"][image_index]
            begin = 0
            #if "has_PGOD" not in sections[pos]:
            #    print(article_id)
            if "captions_has_PGOD" in article and article["captions_has_PGOD"][image_index]:
                nes = article["caption_ner"][image_index]
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
                if "caption_parts_of_speech" in article:
                    words = article["caption_parts_of_speech"][image_index]
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
                n_persons = len(self._get_person_names(
                    article, sample['image_index']))
            else:
                n_persons = 4

            if 'facenet_details' not in sample or n_persons == 0:
                face_embeds = np.array([[]])
            else:
                face_embeds = sample['facenet_details']['embeddings']
                # Keep only the top faces (sorted by size)
                face_embeds = np.array(face_embeds[:n_persons])

            obj_feats = None
            if self.use_objects:
                try:
                    obj = self.db.objects.find_one({'_id': sample['_id']})
                except:
                    os.system("mongod --bind_ip_all --dbpath /home/test/mongodb --wiredTigerCacheSizeGB 10 &")
                    time.sleep(10)
                    continue
                if obj is not None:
                    obj_feats = obj['object_features']
                    if len(obj_feats) == 0:
                        obj_feats = np.array([[]])
                    else:
                        obj_feats = np.array(obj_feats)
                else:
                    obj_feats = np.array([[]])

            yield self.article_to_instance(sample['article_id'], image_count, article_index, 
                                           article, named_entities, face_embeds,
                                           image, fake_captions, sample['image_index'],
                                           image_path, obj_feats)

    def article_to_instance(self, article_id, image_count, article_index, 
                            article, named_entities, face_embeds, image, fake_captions,
                            image_index, image_path, obj_feats) -> Instance:
        context = ' '.join(article['context'].strip().split(' ')[:500])

        caption = article['images'][image_index]
        caption = caption.strip()

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
            "image_index": IndexField(int(image_index), image_filed),
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
                    'web_url': article['web_url'],
                    'image_path': image_path,
                    "article_id": article_id}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def _get_named_entities(self, article):
        # These name indices have the right end point excluded
        names = set()

        if 'context_ner' in article:
            ners = article['context_ner']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, article, pos):
        # These name indices have the right end point excluded
        names = set()

        if 'caption_ner' in article:
            ners = article['caption_ner'][pos]
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

"""Get articles from the New York Times API.

Usage:
    compute_name_statistics.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].

"""
import pickle
from collections import Counter
from datetime import datetime

import ptvsd
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

logger = setup_logger()


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'host': str
    })
    args = schema.validate(args)
    return args


def compute_nytimes(client):
    nytimes = client.nytimes
    start = datetime(2000, 1, 1)
    end = datetime(2019, 5, 1)

    caption_counter = Counter()
    context_counter = Counter()

    article_cursor = nytimes.articles.find({
        'split': 'train',
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        get_proper_names(article['headline'], context_counter)

        sections = article['parsed_section']
        for section in sections:
            if section['type'] == 'caption':
                get_proper_names(section, caption_counter)
            elif section['type'] == 'paragraph':
                get_proper_names(section, context_counter)
            else:
                raise ValueError(f"Unknown type: {section['type']}")

    counters = {
        'caption': caption_counter,
        'context': context_counter,
    }
    with open('./data/nytimes/name_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)


def compute_goodnews(client):
    goodnews = client.goodnews

    caption_counter = Counter()
    context_counter = Counter()

    sample_cursor = goodnews.splits.find(
        {'split': 'train'}, no_cursor_timeout=True).batch_size(128)

    done_article_ids = set()
    for sample in tqdm(sample_cursor):
        if sample['article_id'] in done_article_ids:
            continue
        done_article_ids.add(sample['article_id'])

        article = goodnews.articles.find_one({
            '_id': {'$eq': sample['article_id']},
        })

        get_proper_goodnews_names(
            article, context_counter, 'context_parts_of_speech')

        for idx in article['images'].keys():
            get_caption_proper_names(article, idx, caption_counter)

    counters = {
        'caption': caption_counter,
        'context': context_counter,
    }

    with open('./data/goodnews/name_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)
        

def compute_dailymail(client):
    db = client.dailymail

    caption_counter = Counter()
    context_counter = Counter()

    article_cursor = db.articles.find({
        'split': 'train', 'n_images': {"$lte": 10, "$gte": 1}
    }, no_cursor_timeout=True).batch_size(128)

    for article in tqdm(article_cursor):
        get_proper_names(article['headline'], context_counter)

        sections = article['parsed_section']
        for section in sections:
            if section['type'] == 'caption':
                get_proper_names(section, caption_counter)
            elif section['type'] == 'paragraph':
                get_proper_names(section, context_counter)
            else:
                raise ValueError(f"Unknown type: {section['type']}")

    counters = {
        'caption': caption_counter,
        'context': context_counter,
    }
    with open('/home/test/mic/data/dailymail/name_counters.pkl', 'wb') as f:
        pickle.dump(counters, f)


def get_proper_names(section, counter):
    if 'parts_of_speech' in section:
        parts_of_speech = section['parts_of_speech']
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)


def get_proper_goodnews_names(section, counter, pos_field):
    if pos_field in section:
        parts_of_speech = section[pos_field]
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)


def get_caption_proper_names(article, idx, counter):
    if 'caption_parts_of_speech' in article:
        parts_of_speech = article['caption_parts_of_speech'][idx]
        proper_names = [pos['text'] for pos in parts_of_speech
                        if pos['pos'] == 'PROPN']

        counter.update(proper_names)
        
        
def get_entities(obj):
    nes = obj["named_entities"]
    e = {}
    for ne in nes:
        text = e["text"]
        label = e["label"]
        id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
        if id in e:
            a = e[id]
        else:
            a = {"text":text, "label":label, "_count":0}
            e[id] = a
        a["count"] += 1
    return e


def collect_goodnews_entities(client):
    import pymongo
    db = client.goodnews
    article_cursor = db.articles.find(no_cursor_timeout=True).batch_size(128)
    
    entities = {}
    count = 0
    for article in tqdm(article_cursor):
        
        if "context_ner" in article:
            nes = article["context_ner"]
            for ne in nes:
                text = ne["text"]
                label = ne["label"]
                id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
                if id not in entities:
                    entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
                entities[id]["count"] += 1
                entities[id]["h_count"] += 1
        
        if "caption_ner" in article:
            sample_cursor = db.splits.find({
                'split': {'$eq': "train"}, "article_id": article["_id"]
            }).sort('_id', pymongo.ASCENDING)
            
            for sample in sample_cursor:
                image_index = sample["image_index"]
                nes = article["caption_ner"][image_index]
                
                a = []
                for ne in nes:
                    if ne["label"] in ["PERSON", "GPE", "ORG", "DATE"]:
                        a.append(1)
                if len(a) == 0:
                    count += 1
                
                for ne in nes:
                    text = ne["text"]
                    label = ne["label"]
                    id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
                    if id not in entities:
                        entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
                    entities[id]["count"] += 1
                    entities[id]["c_count"] += 1
            
    print(count)
    for _, entity in entities.items():
        db.entities.insert_one(entity)


def collect_nytimes_entities(client):
    nytimes = client.nytimes
    article_cursor = nytimes.articles.find({
        'split': 'train',
    }, no_cursor_timeout=True).batch_size(128)
    
    entities = {}
    for article in tqdm(article_cursor):
        if "named_entities" not in article['headline']:
            continue
        nes = article['headline']["named_entities"]
        for ne in nes:
            text = ne["text"]
            label = ne["label"]
            id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
            if id not in entities:
                entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
            entities[id]["count"] += 1
            entities[id]["h_count"] += 1
        
        count = 0
        sections = article['parsed_section']
        for section in sections:
            if section['type'] in ['caption', 'paragraph']:
                nes = section["named_entities"]
                if section["text"].strip() =="":
                    continue
                if section['type'] == "caption":
                    a = []
                    for ne in nes:
                        if ne["label"] in ["PERSON", "GPE", "ORG", "DATE"]:
                            a.append(1)
                    if len(a) == 0:
                        count += 1
                for ne in nes:
                    text = ne["text"]
                    label = ne["label"]
                    id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
                    if id not in entities:
                        entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
                    entities[id]["count"] += 1
                    if section['type'] == 'caption':
                        entities[id]["c_count"] += 1
                    else:
                        entities[id]["p_count"] += 1
            else:
                raise ValueError(f"Unknown type: {section['type']}")
            
    print(count)
    for _, entity in entities.items():
        nytimes.entities.insert_one(entity)
        
        
def collect_dailymail_entities(client):
    dailymail = client.dailymail
    article_cursor = dailymail.articles.find({
        'split': 'train', 'n_images': {"$lte": 10, "$gte": 1}
    }, no_cursor_timeout=True).batch_size(128)
    
    entities = {}
    count = 0
    for article in tqdm(article_cursor):
        if "named_entities" not in article['headline']:
            continue
        nes = article['headline']["named_entities"]
        for ne in nes:
            text = ne["text"]
            label = ne["label"]
            id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
            if id not in entities:
                entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
            entities[id]["count"] += 1
            entities[id]["h_count"] += 1
        
        
        sections = article['parsed_section']
        for section in sections:
            if section['type'] in ['caption', 'paragraph']:
                nes = section["named_entities"]
                if section["text"].strip() =="":
                    continue
                if section['type'] == "caption":
                    a = []
                    for ne in nes:
                        if ne["label"] in ["PERSON", "GPE", "ORG", "DATE"]:
                            a.append(1)
                    if len(a) == 0:
                        count += 1
                for ne in nes:
                    text = ne["text"]
                    label = ne["label"]
                    id = text.replace(" ", "_") + "_" + label.replace(" ", "_")
                    if id not in entities:
                        entities[id] = {"_id":id, "text":text, "label":label, "count":0, "h_count": 0, "c_count":0, "p_count":0}
                    entities[id]["count"] += 1
                    if section['type'] == 'caption':
                        entities[id]["c_count"] += 1
                    else:
                        entities[id]["p_count"] += 1
            else:
                raise ValueError(f"Unknown type: {section['type']}")
            
    print(count)
    for _, entity in entities.items():
        dailymail.entities.insert_one(entity)
        
        
def clean_entities(client, db_name):
    import pymongo
    import numpy as np
    if db_name == "dailymail":
        db = client.dailymail
    elif db_name == "nytimes":
        db = client.nytimes
    if db_name == "goodnews":
        db = client.goodnews
    
    article_cursor = db.articles.find(projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(article_cursor)])
    article_cursor.close()
    
    count = 0
    if db_name == "goodnews":
        for _, article_id in enumerate(ids):
            article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
            #print(article_id)
            if 'caption_ner' not in article:
                continue
            captions_ner = article['caption_ner']
            captions = article['images']
            captions_has_PGOD = {}
            article["captions_has_PGOD"] = captions_has_PGOD
            for key in captions:
                if captions[key].strip() =="":
                    captions_has_PGOD[key] = False
                    continue
                
                if len(captions_ner[key]) > 0:
                    nes = captions_ner[key] 
                    a = []
                    for ne in nes:
                        if ne["label"] in ["PERSON", "GPE", "ORG", "DATE"]:
                            a.append(1)
                    if len(a) == 0:
                        captions_has_PGOD[key] = False
                        count += 1
                    else:
                        captions_has_PGOD[key] = True
                else:
                    captions_has_PGOD[key] = False
                            
            db.articles.find_one_and_update({'_id': article['_id']}, {'$set': article})
    else:
        for _, article_id in enumerate(ids):
            article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
            
            sections = article['parsed_section']
            for section in sections:
                if section['type'] == 'caption':
                    
                    if section["text"].strip() =="":
                        section["has_PGOD"] = False
                        continue
                    if "named_entities" in section:
                        nes = section["named_entities"]
                        if section['type'] == "caption":
                            a = []
                            for ne in nes:
                                if ne["label"] in ["PERSON", "GPE", "ORG", "DATE"]:
                                    a.append(1)
                            if len(a) == 0:
                                section["has_PGOD"] = False
                                count += 1
                            else:
                                section["has_PGOD"] = True
                    else:
                        section["has_PGOD"] = False
                            
            db.articles.find_one_and_update({'_id': article['_id']}, {'$set': article})
            
    print(count)
        


def main():
    args = docopt(__doc__, version='0.0.1')
    args = validate(args)

    if args['ptvsd']:
        address = ('0.0.0.0', args['ptvsd'])
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()

    client = MongoClient(host=args['host'], port=27017)

    #compute_nytimes(client)
    #compute_goodnews(client)
    #compute_dailymail(client)
    #collect_nytimes_entities(client)
    #collect_dailymail_entities(client)
    #collect_goodnews_entities(client)
    #clean_entities(client, "nytimes")
    #clean_entities(client, "dailymail")
    clean_entities(client, "goodnews")


if __name__ == '__main__':
    main()

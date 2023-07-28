"""Get articles from the New York Times API.

Usage:
    annotate_dailymail.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -h --host HOST      MongoDB host [default: localhost].
    -b --batch INT      Batch number [default: 1]

"""

import functools
from datetime import datetime
from multiprocessing import Pool

import ptvsd
import spacy
from docopt import docopt
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

from scripts import dailymail
import os, math
from joblib import Parallel, delayed

logger = setup_logger()

def get_parts_of_speech(doc, article):
    parts_of_speech = []
    for tok in doc:
        pos = {
            'start': tok.idx,
            'end': tok.idx + len(tok.text),  # exclude right endpoint
            'text': tok.text,
            'pos': tok.pos_,
        }
        parts_of_speech.append(pos)

        if 'main' in article['headline']:
            section = article['headline']
            assign_pos_to_section(section, pos)

        for section in article['parsed_section']:
            assign_pos_to_section(section, pos)

    article['parts_of_speech'] = parts_of_speech


def assign_pos_to_section(section, pos):
    s = section['spacy_start']
    e = section['spacy_end']
    if pos['start'] >= s and pos['end'] <= e:
        section['parts_of_speech'].append({
            'start': pos['start'] - s,
            'end': pos['end'] - s,
            'text': pos['text'],
            'pos':  pos['pos'],
        })


def calculate_spacy_positions(article):
    title = ''
    cursor = 0
    if 'main' in article['headline']:
        title = article['headline']['main'].strip()
        article['headline']['spacy_start'] = cursor
        cursor += len(title) + 1  # newline
        article['headline']['spacy_end'] = cursor
        article['headline']['parts_of_speech'] = []

    for section in article['parsed_section']:
        text = section['text'].strip()
        section['spacy_start'] = cursor
        cursor += len(text) + 1  # newline
        section['spacy_end'] = cursor
        section['parts_of_speech'] = []


def annotate_pos(article, nlp, db):
    if 'parts_of_speech' in article['parsed_section'][0]:
        return

    calculate_spacy_positions(article)

    title = ''
    if 'main' in article['headline']:
        title = article['headline']['main'].strip()

    sections = article['parsed_section']

    paragraphs = [s['text'].strip() for s in sections]
    paragraphs = [title] + paragraphs

    combined = '\n'.join(paragraphs)

    doc = nlp(combined)
    get_parts_of_speech(doc, article)

    db.articles.find_one_and_update(
        {'_id': article['_id']}, {'$set': article})


def parse_article(article, nlp, db):
    annotate_pos(article, nlp, db)

    sections = article['parsed_section']
    changed = False

    if 'main' in article['headline'] and 'named_entities' not in article['headline']:
        section = article['headline']
        title = section['main'].strip()
        doc = nlp(title)
        section['named_entities'] = []
        for ent in doc.ents:
            changed = True
            ent_info = {
                'start': ent.start_char,
                'end': ent.end_char,
                'text': ent.text,
                'label': ent.label_,
            }
            section['named_entities'].append(ent_info)

    for section in sections:
        if 'named_entities' not in section:
            doc = nlp(section['text'].strip())
            section['named_entities'] = []
            for ent in doc.ents:
                changed = True
                ent_info = {
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'label': ent.label_,
                }
                section['named_entities'].append(ent_info)

    if changed:
        db.articles.find_one_and_update(
            {'_id': article['_id']}, {'$set': article})


def annotate(ids, db):
    nlp = spacy.load("/home/test/mic/softwares/en_core_web_lg-2.1.0")
    for article_id in tqdm(ids):
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
        if article is not None:
            parse_article(article, nlp, db)


def main():
    import pymongo
    import numpy as np
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail
    
    sample_cursor = db.articles.find().sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    sample_cursor.close()
    n_jobs = 12
    #story_files = ["ede107e3337957824c6974e3362e127fc7aff169.story"]
    '''
    batch_size = math.ceil(len(ids) / n_jobs)
    with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
        parallel(delayed(annotate)(ids[batch_size * i : batch_size * (i + 1)], db)
                 for i in range(n_jobs))
    '''
    annotate(ids, db)


if __name__ == '__main__':
    main()













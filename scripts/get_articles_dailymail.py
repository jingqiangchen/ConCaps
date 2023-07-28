# -*- coding: utf-8 -*- 
"""Get articles from the New York Times API.

Usage:
    get_articles.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -r --root-dir DIR   Root directory of data [default: data/nytimes].

"""
import hashlib
import json
import os
import socket
import time
from datetime import datetime
from posixpath import normpath
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import bs4
import ptvsd
import pymongo
import requests
from docopt import docopt
from joblib import Parallel, delayed
from langdetect import detect
from pymongo import MongoClient
from schema import And, Or, Schema, Use
from tqdm import tqdm

from scripts import dailymail
from glob import glob

from tell.utils import setup_logger

import regex as re

logger = setup_logger()


def resolve_url(url):
    """
    resolve_url('http://www.example.com/foo/bar/../../baz/bux/')
    'http://www.example.com/baz/bux/'
    resolve_url('http://www.example.com/some/path/../file.ext')
    'http://www.example.com/some/file.ext'
    """

    parsed = urlparse(url)
    new_path = normpath(parsed.path)
    if parsed.path.endswith('/'):
        # Compensate for issue1707768
        new_path += '/'
    cleaned = parsed._replace(path=new_path)

    return cleaned.geturl()


def get_tags(d, params):
    # See https://stackoverflow.com/a/57683816/3790116
    if any((lambda x: b in x if a == 'class' else b == x)(d.attrs.get(a, [])) for a, b in params.get(d.name, {}).items()):
        yield d
    for i in filter(lambda x: x != '\n' and not isinstance(x, bs4.element.NavigableString), d.contents):
        yield from get_tags(i, params)


def extract_text_new(soup):
    # For articles between 2013 and 2019
    sections = []
    article_node = soup.find('article')

    params = {
        'div': {'class': 'StoryBodyCompanionColumn'},
        'figcaption': {'itemprop': 'caption description'},
        'figure': {'class': 'e1g7ppur0'},
    }

    article_parts = get_tags(article_node, params)
    i = 0

    for part in article_parts:
        has_caption = False
        if part.name == 'div':
            paragraphs = part.find_all(['p', 'h2'])
            for p in paragraphs:
                sections.append({'type': 'paragraph', 'text': p.text.strip()})
        elif part.name == 'figcaption':
            if part.parent.attrs.get('itemid', 0):
                caption = part.find('span', {'class': 'e13ogyst0'})
                raw_url = part.parent.attrs['itemid']
                has_caption = True
        elif part.name == 'figure':
            if part.attrs.get('itemid', 0):
                caption = part.find('span', {'class': 'e13ogyst0'})
                raw_url = part.attrs['itemid']
                has_caption = True

        if has_caption and caption:
            url = resolve_url(raw_url)
            sections.append({
                'type': 'caption',
                'order': i,
                'text': caption.text.strip(),
                'url': url,
                'hash': hashlib.sha256(url.encode('utf-8')).hexdigest(),
            })
            i += 1

    return sections


def extract_text_old(soup):
    # For articles in 2012 and earlier
    sections = []

    params = {
        'p': {'class': 'story-body-text'},
        'figcaption': {'itemprop': 'caption description'},
        'span': {'class': 'caption-text'},
    }

    article_parts = get_tags(soup, params)
    i = 0
    for part in article_parts:
        if part.name == 'p':
            sections.append({'type': 'paragraph', 'text': part.text.strip()})
        elif part.name == 'figcaption':
            if part.parent.attrs.get('itemid', 0):
                caption = part.find('span', {'class': 'caption-text'})
                if caption:
                    url = resolve_url(part.parent.attrs['itemid'])
                    sections.append({
                        'type': 'caption',
                        'order': i,
                        'text': caption.text.strip(),
                        'url': url,
                        'hash': hashlib.sha256(url.encode('utf-8')).hexdigest(),
                    })
                    i += 1

    return sections


def extract_text(html):
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # Newer articles use StoryBodyCompanionColumn
    if soup.find('article') and soup.find('article').find_all('div', {'class': 'StoryBodyCompanionColumn'}):
        return extract_text_new(soup)

    # Older articles use story-body
    elif soup.find_all('p', {'class': 'story-body-text'}):
        return extract_text_old(soup)

    return []


def retrieve_article(article, root_dir, db):
    if article['_id'].startswith('nyt://article/'):
        article['_id'] = article['_id'][14:]

    result = db.source.find_one({'_id': article['_id']})
    if result is not None:
        return

    data = article
    data['scraped'] = False
    data['parsed'] = False
    data['error'] = False
    data['pub_date'] = None

    if not article['web_url']:
        return

    url = resolve_url(article['web_url'])
    for i in range(10):
        try:
            response = urlopen(url, timeout=20)
            break
        except (ValueError, HTTPError):
            # ValueError: unknown url type: '/interactive/2018/12/05/business/05Markets.html'
            # urllib.error.HTTPError: HTTP Error 404: Not Found
            return
        except (URLError, ConnectionResetError):
            time.sleep(60)
            continue
        except socket.timeout:
            pass
        # urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
        return

    data['web_url'] = url
    try:
        raw_html = response.read().decode('utf-8')
    except UnicodeDecodeError:
        return

    raw_data = {
        '_id': article['_id'],
        'raw_html': raw_html,
    }

    parsed_sections = extract_text(raw_html)
    data['parsed_section'] = parsed_sections

    text_list = [sec['text'] for sec in article['parsed_section']]
    text = '\n'.join(text_list)
    data['language'] = detect(text)

    if parsed_sections:
        image_positions = []
        for i, section in enumerate(parsed_sections):
            if section['type'] == 'caption':
                image_positions.append(i)
                img_path = os.path.join(root_dir, 'images',
                                        f"{section['hash']}.jpg")
                if not os.path.exists(img_path):
                    try:
                        img_response = requests.get(
                            section['url'], stream=True)
                        img_data = img_response.content
                        with open(img_path, 'wb') as f:
                            f.write(img_data)

                        db.images.update_one(
                            {'_id': section['hash']},
                            {'$push': {'captions': {
                                'id': article['_id'],
                                'caption': section['text'],
                            }}}, upsert=True)

                    except requests.exceptions.MissingSchema:
                        section['downloaded'] = False
                    else:
                        section['downloaded'] = True

        data['parsed'] = True
        article['image_positions'] = image_positions
        article['n_images'] = len(image_positions)
    else:
        article['n_images'] = 0

    data['scraped'] = True

    db.source.insert_one({'_id': raw_data['_id']}, {'$set': raw_data})

    if not article['parsed'] or article['n_images'] == 0 or article['language'] != 'en':
        pass #db.text_articles.insert_one(article)
    else:
        pass #db.articles.insert_one({'_id': article['_id']}, {'$set': data})


def extract_titles():
    import codecs
    def re_version():
        wf = open(dailymail.path_titles, "w", encoding="utf-8")
        files = os.listdir(dailymail.path_htmls)
        count = 0
        for file in files:
            sign = False
            with codecs.open(dailymail.path_htmls + "/" + file, encoding="utf8", errors="ignore") as f:
                html = f.read()
                html = html.replace("\n", "")
                #print(file)
                m = re.search(r"<title>(.+?)</title>", html)
                if m:
                    title = m.group(1)
                    title = title[:title.rindex("|")].strip()
                    wf.write(os.path.splitext(file)[0] + "\n")
                    wf.write(title + "\n")
                    sign = True
                    count += 1
                    #print(title[:title.rindex("|")].strip())
                    continue
            if not sign:
                print(file, count)
        wf.close()
        
    def soup_version():
        pass
    
    re_version()

def retrieve_articles(files, titles, db): # 40
    # _id abstract source document_type word_count image_positions parsed_section n_image language 
    # parts_of_speech n_images_with_faces split 
    
    for file in files:
        article = {}
        captions = {}
        
        first, _ = os.path.splitext(file)
        article["_id"] = first
        result = db.articles.find_one({'_id': {"$eq": article['_id']}})
        if result is not None:
            #print(first)
            continue
        
        article["source"] = "Daily Mail"
        article["document_type"] = "article"
        article["language"] = "en"
        article["headline"] = {"main": titles[first]}
        
        if not os.path.exists(dailymail.path_captions_bak + "/" +first):
            continue
        #print(file)
        with open(dailymail.path_captions_bak + "/" +first) as f:
            tmp_caps = []
            tmp_cap_nos = []
            lines = f.readlines()
            
            if len(lines) <= 1:
                continue
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line == "": 
                    continue
                if re.match(r"^[\d ]+$", line):
                    #print(line)
                    items = line.split(" ") 
                    items = [int(item) for item in items]
                    tmp_cap_nos.append(items)
                    tmp_caps.append([])
                elif len(tmp_caps) > 0:
                    tmp_caps[-1].append(line)
            for no, cap in zip(tmp_cap_nos, tmp_caps):
                caption = " ".join(cap)
                if caption not in captions:
                    captions[caption] = []
                captions[caption].extend(no)
        
        parsed_section = []
        article["parsed_section"] = parsed_section
        image_positions = []
        article["image_positions"] = image_positions
        abstract = []
        article["abstract"] = abstract
        para_index = 0
        with open(dailymail.path_stories + "/" + file) as f:
            is_highlight = False
            text_lines = []
            p = {"type":"paragraph"}
            for line in f.readlines():
                line = line.strip()
                if line == "" and len(text_lines) > 0:
                    if is_highlight:
                        abstract.append(" ".join(text_lines))
                        text_lines = []
                    else:
                        text = " ".join(text_lines)
                        p["text"] = text
                        if text in captions: # image
                            p["type"] = "caption"
                            if len(captions[text]) > 1:t=True
                            for image_index in captions[text]:
                                pp = p.copy()
                                pp["hash"] = first + "-" + str(image_index)
                                parsed_section.append(pp)
                                image_positions.append(para_index)
                                para_index += 1
                            del captions[text]
                        else:
                            parsed_section.append(p)
                            para_index += 1
                        text_lines = []
                        p = {"type":"paragraph"}
                elif line == "@highlight": # abstract
                    is_highlight = True
                else: # paragraph
                    text_lines.append(line)
            else:
                if line != "":
                    if is_highlight:
                        abstract.append(" ".join(text_lines))
                    else:
                        p["text"] = " ".join(text_lines)
                        parsed_section.append(p)
        
        
        article["n_images"] = len(image_positions)
        article["parsed"] = False
        #db.articles.delete_one({'_id': article['_id']})
        #db.articles.insert_one({'_id': article['_id']}, {'$set': article})
        db.articles.insert_one(article)
        
        for image_position in image_positions:
            section = parsed_section[image_position]
            try:
                db.images.insert_one( {'_id': section['hash'],
                                   'captions': [{'id': article['_id'], 'caption': section['text']}]})
            except:
                print(section['hash'])
        
    # db.scraping.insert_one({'year': year, 'month': month})


def create_db():
    import math
    #root_dir = dailymail.path_images
    img_dir = dailymail.path_images #os.path.join(root_dir, 'images')
    #os.makedirs(img_dir, exist_ok=True)

    # Get the nytimes database
    client = MongoClient(host='localhost', port=27017)
    db = client.dailymail
    
    #db.articles.delete_many({})
    #db.images.delete_many({})
    
    jobs = 12
    #story_files = glob(f'{dailymail.path_stories}/*')
    story_files = os.listdir(dailymail.path_stories)
    story_files.sort()
    story_files = story_files
    #story_files = ["ede107e3337957824c6974e3362e127fc7aff169.story"]
    batch_size = math.ceil(len(story_files) / jobs)
    
    titles = {}
    with open(dailymail.path_titles, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f.readlines()):
            if i % 2 == 0:
                key = line.strip()
            else:
                titles[key] = line.strip()
    
    with Parallel(n_jobs=jobs, backend='threading') as parallel:
        parallel(delayed(retrieve_articles)(story_files[batch_size * i : batch_size * (i + 1)], titles, db)
                 for i in range(jobs))

def create_indices():
    client = MongoClient(host='localhost', port=27017)
    db = client.dailymail
    
    wf = open(dailymail.path_corpus + "/train-error.txt", "w")
    with open(dailymail.path_trains) as f:
        for line in f.readlines():
            line = line.strip()
            article = db.articles.find_one({
                '_id': {'$eq': line},
            })
            if article is None:
                print(line)
                wf.write(line + "\n")
                continue
            
            article['split'] = 'train'
            db.articles.find_one_and_update(
                    {'_id': article['_id']}, {'$set': article})
    wf.close()
    
    wf = open(dailymail.path_corpus + "/dev-error.txt", "w")
    with open(dailymail.path_devs) as f:
        for line in f.readlines():
            line = line.strip()
            article = db.articles.find_one({
                '_id': {'$eq': line},
            })
            if article is None:
                print(line)
                wf.write(line + "\n")
                continue
            
            article['split'] = 'valid'
            db.articles.find_one_and_update(
                    {'_id': article['_id']}, {'$set': article})
    wf.close()
    
    wf = open(dailymail.path_corpus + "/test-error.txt", "w")
    with open(dailymail.path_tests) as f:
        for line in f.readlines():
            line = line.strip()
            article = db.articles.find_one({
                '_id': {'$eq': line},
            })
            if article is None:
                print(line)
                wf.write(line + "\n")
                continue
            
            article['split'] = 'test'
            db.articles.find_one_and_update(
                    {'_id': article['_id']}, {'$set': article})
    wf.close()

    # Build indices
    logger.info('Building indices')

    db.articles.create_index([
        ('_id', pymongo.ASCENDING),
    ])

    db.articles.create_index([
        ('split', pymongo.ASCENDING),
        ('_id', pymongo.ASCENDING),
    ])

    db.articles.create_index([
        ('n_images', pymongo.ASCENDING),
        ('n_images_with_faces', pymongo.ASCENDING),
        ('pub_date', pymongo.DESCENDING),
    ])
    
    
def main():
    #create_db()
    create_indices()
    

def test():
    import numpy as np
    client = MongoClient(host='localhost', port=27017)
    db = client.dailymail
    sample_cursor = db.articles.find({
            'split': "valid",
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    count_4 = 0
    count_5 = 0
    all = 0
    for i, article_id in enumerate(ids):
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
        all += article["n_images"]
        if i < len(ids) // 4:
            count_4 += article["n_images"]
        if i < len(ids) // 5:
            count_5 += article["n_images"]
    print(all/len(ids), count_4/(len(ids)//4), count_5/(len(ids)//5))

if __name__ == '__main__':
    #main()
    test()
    #extract_titles()












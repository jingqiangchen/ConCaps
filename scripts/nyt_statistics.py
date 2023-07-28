import logging
import os
import random
import re
from typing import Dict

import numpy as np
import pymongo
import torch

from PIL import Image
from pymongo import MongoClient
from tqdm import tqdm

import sys, time, json

from pycocoevalcap.rouge.rouge import Rouge

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class NYTStatistics:

    def __init__(self,
                 image_dir: str = "/home/test/images_processed",
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017) -> None:
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.nytimes
        self.image_dir = image_dir

        self.rouge_scorer = Rouge()
        
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

    def compute(self, split: str):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')
        
        err_file = open("nyt_err.txt", "w")
        out_file = open("nyt_out.txt", "w")
        #sys.stderr = err_file
        #sys.stdout = out_file
        
        pre_out = open("nyt_pre_out.txt", "w")
        pre_out2 = open("nyt_pre_out2.txt", "w")
        pre_out3 = open("nyt_pre_out3.txt", "w")
        pre_out4 = open("nyt_pre_out4.txt", "w")
        pre_out5 = open("nyt_pre_out5.txt", "w")
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
        
        text_scores5 = []
        text_scores4 = []
        text_scores3 = []
        text_scores2 = []
        text_scores = []
        caption_scores5 = []
        caption_scores4 = []
        caption_scores3 = []
        caption_scores2 = []
        caption_scores = []
        for article_index, article_id in enumerate(ids):
            if article_id == '58a4cf7795d0e02474636b8c':
                continue
            try:
                article = self.db.articles.find_one(
                    {'_id': {'$eq': article_id}}, projection=projection)
            except:
                err_file.write(article_id + "\n")
                os.system("mongod --bind_ip_all --dbpath /home/test/mongodb --wiredTigerCacheSizeGB 10 &")
                time.sleep(10)
                continue
            sections = article['parsed_section']
            image_positions = article['image_positions']
            
            pre_text5 = None
            pre_text4 = None
            pre_text3 = None
            pre_text2 = None
            pre_text = None
            cur_text = None
            
            pre_caption5 = None
            pre_caption4 = None
            pre_caption3 = None
            pre_caption2 = None
            pre_caption = None
            
            pre_caption_named_entities5 = set()
            pre_caption_named_entities4 = set()
            pre_caption_named_entities3 = set()
            pre_caption_named_entities2 = set()
            pre_caption_named_entities = set()
            
            cur_caption = None
            
            for pos in image_positions:
                caption = sections[pos]['text'].strip()
                if not caption:
                    continue
                
                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue
                
                title = ''
                if 'main' in article['headline']:
                    title = article['headline']['main'].strip()
                paragraphs = []
                named_entities = set()
                caption_named_entities = set()
                n_words = 0
                if title:
                    paragraphs.append(title)
                    named_entities.union(
                        self._get_named_entities(article['headline']))
                    n_words += len(self.to_token_ids(title))

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
                
                paragraphs = paragraphs + before + after
                named_entities = sorted(named_entities)
                
                cur_text = '\n'.join(paragraphs).strip()
                cur_caption = caption
                caption_named_entities = self._get_named_entities(sections[pos])
                cur_caption_named_entities = caption_named_entities
                if pre_text is not None and pre_caption is not None:
                    text_rouge1 = self.rouge_scorer.calc_score([cur_text], [pre_text])
                    text_rouge2 = self.rouge_scorer.calc_score([pre_text], [cur_text])
                    text_scores.append((text_rouge1 + text_rouge2) / 2)
                    
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption], [cur_caption])
                    caption_scores.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities & cur_caption_named_entities)
                    
                    pre_out.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")
                    
                if pre_text2 is not None and pre_caption2 is not None:
                    text_rouge1 = self.rouge_scorer.calc_score([cur_text], [pre_text2])
                    text_rouge2 = self.rouge_scorer.calc_score([pre_text2], [cur_text])
                    text_scores2.append((text_rouge1 + text_rouge2) / 2)
                    
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption2])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption2], [cur_caption])
                    caption_scores2.append((caption_rouge1 + caption_rouge2) / 2)
                    #pre_out2.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\n")
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities2)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities2 & cur_caption_named_entities)
                    
                    pre_out2.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")
                    
                if pre_text3 is not None and pre_caption3 is not None:
                    text_rouge1 = self.rouge_scorer.calc_score([cur_text], [pre_text3])
                    text_rouge2 = self.rouge_scorer.calc_score([pre_text3], [cur_text])
                    text_scores3.append((text_rouge1 + text_rouge2) / 2)
                    
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption3])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption3], [cur_caption])
                    caption_scores3.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities3)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities3 & cur_caption_named_entities)
                    
                    pre_out3.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")
                    
                if pre_text4 is not None and pre_caption4 is not None:
                    text_rouge1 = self.rouge_scorer.calc_score([cur_text], [pre_text4])
                    text_rouge2 = self.rouge_scorer.calc_score([pre_text4], [cur_text])
                    text_scores4.append((text_rouge1 + text_rouge2) / 2)
                    
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption4])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption4], [cur_caption])
                    caption_scores4.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities4)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities4 & cur_caption_named_entities)
                    
                    pre_out4.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")
                    
                if pre_text5 is not None and pre_caption5 is not None:
                    text_rouge1 = self.rouge_scorer.calc_score([cur_text], [pre_text5])
                    text_rouge2 = self.rouge_scorer.calc_score([pre_text5], [cur_text])
                    text_scores5.append((text_rouge1 + text_rouge2) / 2)
                    
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption5])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption5], [cur_caption])
                    caption_scores5.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities5)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities5 & cur_caption_named_entities)
                    
                    pre_out5.write(f"{text_rouge1}\t{text_rouge2}\t{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")
                
                pre_text5 = pre_text4
                pre_caption5 = pre_caption4
                pre_caption_named_entities5 = pre_caption_named_entities4
                pre_text4 = pre_text3
                pre_caption4 = pre_caption3
                pre_caption_named_entities4 = pre_caption_named_entities3
                pre_text3 = pre_text2
                pre_caption3 = pre_caption2
                pre_caption_named_entities3 = pre_caption_named_entities2
                pre_text2 = pre_text
                pre_caption2 = pre_caption
                pre_caption_named_entities2 = pre_caption_named_entities
                pre_text = cur_text
                pre_caption = cur_caption
                pre_caption_named_entities = cur_caption_named_entities
            
            pre_out.write("\n")
            pre_out2.write("\n")
            pre_out3.write("\n")
            pre_out4.write("\n")
            pre_out5.write("\n")
        
        pre_out.close()
        pre_out2.close()
        pre_out3.close()
        pre_out4.close()
        pre_out5.close()
        
        mean_text_score = np.mean(np.array(text_scores))
        mean_text_score2 = np.mean(np.array(text_scores2))
        mean_text_score3 = np.mean(np.array(text_scores3))
        mean_text_score4 = np.mean(np.array(text_scores4))
        mean_text_score5 = np.mean(np.array(text_scores5))
        
        mean_caption_score = np.mean(np.array(caption_scores))
        mean_caption_score2 = np.mean(np.array(caption_scores2))
        mean_caption_score3 = np.mean(np.array(caption_scores3))
        mean_caption_score4 = np.mean(np.array(caption_scores4))
        mean_caption_score5 = np.mean(np.array(caption_scores5))
        
        print(mean_text_score, mean_caption_score)
        print(mean_text_score2, mean_caption_score2)
        print(mean_text_score3, mean_caption_score3)
        print(mean_text_score4, mean_caption_score4)
        print(mean_text_score5, mean_caption_score5)
        
        err_file.close()
        out_file.close()
        
    def compute1(self):
        pre_out = np.loadtxt("nyt_new_pre_out.txt", delimiter="\t")
        pre_out2 = np.loadtxt("nyt_new_pre_out2.txt", delimiter="\t")
        pre_out3 = np.loadtxt("nyt_new_pre_out3.txt", delimiter="\t")
        pre_out4 = np.loadtxt("nyt_new_pre_out4.txt", delimiter="\t")
        pre_out5 = np.loadtxt("nyt_new_pre_out5.txt", delimiter="\t")
        
        text_score1 = np.mean(pre_out[:, :2])
        caption_score1 = np.mean(pre_out[:, 2:4])
        text_score2 = np.mean(pre_out2[:, :2])
        caption_score2 = np.mean(pre_out2[:, 2:4])
        text_score3 = np.mean(pre_out3[:, :2])
        caption_score3 = np.mean(pre_out3[:, 2:4])
        text_score4 = np.mean(pre_out4[:, :2])
        caption_score4 = np.mean(pre_out4[:, 2:4])
        text_score5 = np.mean(pre_out5[:, :2])
        caption_score5 = np.mean(pre_out5[:, 2:4])
        
        pre_outs = [pre_out, pre_out2, pre_out3, pre_out4, pre_out5]
        entity_scores = []
        for pre_out in pre_outs:
            entity_score1 = pre_out[:, 4]
            entity_score1 = np.where(entity_score1==0, 1, entity_score1)
            entity_score2 = pre_out[:, 5]
            entity_score2 = np.where(entity_score2==0, 1, entity_score2)
            entity_score3 = pre_out[:, 6]
            entity_score1 = entity_score3 / entity_score1
            entity_score2 = entity_score3 / entity_score2
            entity_score3 = entity_score1 + entity_score2
            entity_score3 = np.where(entity_score3 == 0, 1, entity_score3)
            entity_score3 = 2 * entity_score1 * entity_score2 / entity_score3
            entity_score = np.mean(entity_score3)
            entity_scores.append(entity_score)
        
        with open("nyt_compute1.txt", "w") as f:
            f.write(f"{text_score1}\t{caption_score1}\t{entity_scores[0]}\n") 
            f.write(f"{text_score2}\t{caption_score2}\t{entity_scores[1]}\n") 
            f.write(f"{text_score3}\t{caption_score3}\t{entity_scores[2]}\n") 
            f.write(f"{text_score4}\t{caption_score4}\t{entity_scores[3]}\n") 
            f.write(f"{text_score5}\t{caption_score5}\t{entity_scores[4]}\n") 
        
    def compute2(self):
        pre_out = np.loadtxt("new_pre_out.txt", delimiter="\t")
        pre_out2 = np.loadtxt("new_pre_out2.txt", delimiter="\t")
        pre_out3 = np.loadtxt("new_pre_out3.txt", delimiter="\t")
        pre_out4 = np.loadtxt("new_pre_out4.txt", delimiter="\t")
        pre_out5 = np.loadtxt("new_pre_out5.txt", delimiter="\t")
        
        pre_out_mean = np.mean(pre_out, axis=0)
        pre_out_mean2 = np.mean(pre_out2, axis=0)
        pre_out_mean3 = np.mean(pre_out3, axis=0)
        pre_out_mean4 = np.mean(pre_out4, axis=0)
        pre_out_mean5 = np.mean(pre_out5, axis=0)
        
        print(pre_out_mean)
        print(pre_out_mean2)
        print(pre_out_mean3)
        print(pre_out_mean4)
        print(pre_out_mean5)
        
    def compute3(self):
        out_file = open("nyt_compute3.txt", "w")
        sys.stdout = out_file
        
        pre_out = np.loadtxt("nyt_new_pre_out.txt", delimiter="\t")
        a1 = np.mean(pre_out[:, :2], axis=1)
        a2 = np.mean(pre_out[:, 2:4], axis=1)
        a = np.stack([a1, a2], axis=1).tolist()
        a.sort(key=lambda x:[-x[0], -x[1]])
        a = np.array(a)
        M = len(a)
        for i in range(1, 11):
            j_1 = int(M / 10 * (i - 1))
            j = int(M / 10 * i)
            b = np.mean(a[:j, :], axis=0)
            print(i, b)
            
        out_file.close()
            
    def compute4(self):
        pre_out = np.loadtxt("/home/test/mic/codes/transfrom-and-tell-1120/pre_out.txt", delimiter="\t")
        a1 = np.mean(pre_out[:, :2], axis=1)
        a2 = np.mean(pre_out[:, 5:8], axis=1)
        
        a = np.stack([a1, a2], axis=1).tolist()
        a.sort(key=lambda x:-x)
        a = np.array(a)
        M = len(a)
        for i in range(1, 11):
            j_1 = int(M / 10 * (i - 1))
            j = int(M / 10 * i)
            b = np.mean(a[:j, :], axis=0)
            print(i, b)
    
    def compute_for_generations(self, gen_path, out_path):
        err_file = open("err.txt", "w")
        out_file = open("out.txt", "w")
        sys.stderr = err_file
        sys.stdout = out_file
        
        gen_captions = {}
        with open(gen_path) as f:
            for line in f.readlines():
                obj = json.loads(line)
                image_id = os.path.splitext(os.path.basename(obj["image_path"]))[0] 
                gen_captions[image_id] = obj
                #gen_captions[image_id] = obj["generation"] 
                #gen_entities[image_id] = obj["generated_entities"] 
        
        pre_out = open(out_path, "w")
        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({
            'split': "test",
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)]) 
        sample_cursor.close()
        self.rs.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']

        caption_scores = []
        for article_index, article_id in enumerate(ids):
            
            article = self.db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']
            image_positions = article['image_positions']

            pre_caption = None
            pre_caption_named_entities = None
            cur_caption = None
            
            for pos in image_positions:
                caption = sections[pos]['text'].strip()
                if not caption:
                    continue
                
                image_id = sections[pos]['hash']
                
                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue
                
                if image_id not in gen_captions:
                    print(image_id)
                    continue
                obj = gen_captions[image_id]
                
                cur_caption = obj["generation"]
                cur_caption_named_entities = self._get_generation_entities(obj)
                
                if pre_caption is not None and pre_caption_named_entities is not None:
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption], [cur_caption])
                    caption_scores.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities & cur_caption_named_entities)
                    #print(pre_caption_named_entities, cur_caption_named_entities)
                    pre_out.write(f"{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")

                pre_caption = cur_caption
                pre_caption_named_entities = cur_caption_named_entities
            
            pre_out.write("\n")
        
        err_file.close()
        out_file.close()
        
    def compute1_generation(self, gen_path, out_path):
        pre_out = np.loadtxt(gen_path, delimiter="\t")
        
        caption_score1 = np.mean(pre_out[:, :2])
        
        entity_score1 = pre_out[:, 2]
        entity_score1 = np.where(entity_score1==0, 1, entity_score1)
        entity_score2 = pre_out[:, 3]
        entity_score2 = np.where(entity_score2==0, 1, entity_score2)
        entity_score3 = pre_out[:, 4]
        entity_score1 = entity_score3 / entity_score1
        entity_score2 = entity_score3 / entity_score2
        entity_score3 = entity_score1 + entity_score2
        entity_score3 = np.where(entity_score3 == 0, 1, entity_score3)
        entity_score3 = 2 * entity_score1 * entity_score2 / entity_score3
        entity_score = np.mean(entity_score3)
        
        with open(out_path, "w") as f:
            f.write(f"{caption_score1}\t{entity_score}\n")

    def compute_for_got(self, out_path):
        err_file = open("err.txt", "w")
        out_file = open("out.txt", "w")
        sys.stderr = err_file
        sys.stdout = out_file
        
        pre_out = open(out_path, "w")
        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.articles.find({
            'split': "test",
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)]) 
        sample_cursor.close()
        self.rs.shuffle(ids)

        projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']

        caption_scores = []
        for article_index, article_id in enumerate(ids):
            
            article = self.db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
            sections = article['parsed_section']
            image_positions = article['image_positions']

            pre_caption = None
            pre_caption_named_entities = None
            cur_caption = None
            
            for pos in image_positions:
                caption = sections[pos]['text'].strip()
                if not caption:
                    continue
                
                image_id = sections[pos]['hash']
                
                image_path = os.path.join(
                    self.image_dir, f"{sections[pos]['hash']}.jpg")
                try:
                    image = Image.open(image_path)
                except (FileNotFoundError, OSError):
                    continue
                
                cur_caption = caption
                cur_caption_named_entities = self._get_named_entities(sections[pos]) 
                
                if pre_caption is not None and pre_caption_named_entities is not None:
                    caption_rouge1 = self.rouge_scorer.calc_score([cur_caption], [pre_caption])
                    caption_rouge2 = self.rouge_scorer.calc_score([pre_caption], [cur_caption])
                    caption_scores.append((caption_rouge1 + caption_rouge2) / 2)
                    
                    caption_named_entities_score1 = len(pre_caption_named_entities)
                    caption_named_entities_score2 = len(cur_caption_named_entities)
                    caption_named_entities_score3 = len(pre_caption_named_entities & cur_caption_named_entities)
                    #print(pre_caption_named_entities, cur_caption_named_entities)
                    pre_out.write(f"{caption_rouge1}\t{caption_rouge2}\t{caption_named_entities_score1}\t{caption_named_entities_score2}\t{caption_named_entities_score3}\n")

                pre_caption = cur_caption
                pre_caption_named_entities = cur_caption_named_entities
            
            pre_out.write("\n")
        
        err_file.close()
        out_file.close()
        
    def compute1_got(self, gen_path, out_path):
        pre_out = np.loadtxt(gen_path, delimiter="\t")
        
        caption_score1 = np.mean(pre_out[:, :2])
        
        entity_score1 = pre_out[:, 2]
        entity_score1 = np.where(entity_score1==0, 1, entity_score1)
        entity_score2 = pre_out[:, 3]
        entity_score2 = np.where(entity_score2==0, 1, entity_score2)
        entity_score3 = pre_out[:, 4]
        entity_score1 = entity_score3 / entity_score1
        entity_score2 = entity_score3 / entity_score2
        entity_score3 = entity_score1 + entity_score2
        entity_score3 = np.where(entity_score3 == 0, 1, entity_score3)
        entity_score3 = 2 * entity_score1 * entity_score2 / entity_score3
        entity_score = np.mean(entity_score3)
        
        with open(out_path, "w") as f:
            f.write(f"{caption_score1}\t{entity_score}\n")

    def _get_named_entities(self, section):
        # These name indices have the right end point excluded
        names = set()

        if 'named_entities' in section:
            ners = section['named_entities']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names
    
    def _get_generation_entities(self, section):
        # These name indices have the right end point excluded
        names = set()
        
        if 'generated_entities' in section:
            ners = section['generated_entities']
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
    
def postprocess():
    pre_in = open("nyt_pre_out.txt")
    pre_out = open("nyt_new_pre_out.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    i = 0
    pre_in = open("nyt_pre_out2.txt")
    pre_out = open("nyt_new_pre_out2.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            if i % 2 == 1:
                pre_out.write(line + "\n")
            i += 1
    pre_in.close()
    pre_out.close()
    
    pre_in = open("nyt_pre_out3.txt")
    pre_out = open("nyt_new_pre_out3.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("nyt_pre_out4.txt")
    pre_out = open("nyt_new_pre_out4.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("nyt_pre_out5.txt")
    pre_out = open("nyt_new_pre_out5.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()

if __name__ == '__main__':
    #NYTStatistics().compute("train")
    #postprocess()
    #NYTStatistics().compute1()
    #NYTStatistics().compute3()

    #NYTStatistics().compute_for_generations("/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_fc2/serialization/generations.jsonl", "NYT_MMM_fc2.txt")
    #NYTStatistics().compute1_generation("NYT_MMM_fc2.txt", "NYT_MMM_fc2_out.txt")
    #NYTStatistics().compute_for_generations("/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_fd1_v2/serialization/generations.jsonl", "NYT_MMM_fd1_v2.txt")
    #NYTStatistics().compute1_generation("NYT_MMM_fd1_v2.txt", "NYT_MMM_fd1_v2_out.txt")
    #NYTStatistics().compute_for_generations("/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_fcd1_v2/serialization/generations.jsonl", "NYT_MMM_fcd1_v2.txt")
    #NYTStatistics().compute1_generation("NYT_MMM_fcd1_v2.txt", "NYT_MMM_fcd1_v2_out.txt")
    #NYTStatistics().compute_for_generations("/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/9_transformer_objects/serialization/generations.jsonl", "NYT_9_transformer_objects.txt")
    #NYTStatistics().compute1_generation("NYT_9_transformer_objects.txt", "NYT_9_transformer_objects_out.txt")
    #NYTStatistics().compute_for_generations("/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_f1/serialization/generations.jsonl", "NYT_MMM_f1.txt")
    #NYTStatistics().compute1_generation("NYT_MMM_f1.txt", "NYT_MMM_f1_out.txt")
    #NYTStatistics().compute_for_generations("/home/test/mic/codes/show-attend-tell/nytimes/generations.json", "nyt_show-attend-tell.txt")
    NYTStatistics().compute1_generation("nyt_show-attend-tell.txt", "nyt_show-attend-tell_out.txt")
    #NYTStatistics().compute_for_got("nytimes_got.txt")
    #NYTStatistics().compute1_generation("nytimes_got.txt", "nytimes_got_out.txt")



























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

import sys, time

from pycocoevalcap.rouge.rouge import Rouge

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class GoodNewsStatistics:

    def __init__(self,
                 image_dir: str = "/home/test/images_processed_goodnews",
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017) -> None:
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.goodnews
        self.image_dir = image_dir

        self.rouge_scorer = Rouge()
        
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

    def compute(self, split: str):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        err_file = open("err.txt", "w")
        out_file = open("out.txt", "w")
        #sys.stderr = err_file
        #sys.stdout = out_file
        
        pre_out = open("goodnews_pre_out.txt", "w")
        pre_out2 = open("goodnews_pre_out2.txt", "w")
        pre_out3 = open("goodnews_pre_out3.txt", "w")
        pre_out4 = open("goodnews_pre_out4.txt", "w")
        pre_out5 = open("goodnews_pre_out5.txt", "w") 

        limit = self.eval_limit if split == 'val' else 0

        logger.info('Grabbing all article IDs')
        sample_cursor = self.db.splits.find({
            'split': {'$eq': split},
        }, projection=['_id'], limit=limit).sort('_id', pymongo.ASCENDING)

        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        #self.rs.shuffle(ids)
        
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
        
        article_index = 0
        pre_article_id = ""
        image_count = 0
        for sample_id in ids:
            sample = self.db.splits.find_one({'_id': {'$eq': sample_id}})

            # Find the corresponding article
            article = self.db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            }, projection=['_id', 'context', 'images', 'web_url', 'caption_ner', 'context_ner', "captions_has_PGOD"])

            # Load the image
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue
            
            image_index = sample["image_index"]
            #print(sample["article_id"], pre_article_id)
            if sample["article_id"] != pre_article_id:
                article_index += 1
                pre_article_id = sample["article_id"]
                image_count = 0
                
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
            
            #named_entities = sorted(self._get_named_entities(article))
            cur_text = ' '.join(article['context'].strip().split(' ')[:500])
            caption = article["images"][image_index] 
            cur_caption = caption
            caption_named_entities = self._get_caption_named_entities(article['caption_ner'][image_index])
            
            cur_caption_named_entities = caption_named_entities
            #print(cur_caption_named_entities, pre_caption_named_entities) 
            #print(article['caption_ner'][image_index])
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
        pre_out = np.loadtxt("goodnews_new_pre_out.txt", delimiter="\t")
        pre_out2 = np.loadtxt("goodnews_new_pre_out2.txt", delimiter="\t")
        pre_out3 = np.loadtxt("goodnews_new_pre_out3.txt", delimiter="\t")
        pre_out4 = np.loadtxt("goodnews_new_pre_out4.txt", delimiter="\t")
        pre_out5 = np.loadtxt("goodnews_new_pre_out5.txt", delimiter="\t")
        
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
        
        with open("goodnews_compute1.txt", "w") as f:
            f.write(f"{text_score1}\t{caption_score1}\t{entity_scores[0]}\n")
            f.write(f"{text_score2}\t{caption_score2}\t{entity_scores[1]}\n")
            f.write(f"{text_score3}\t{caption_score3}\t{entity_scores[2]}\n")
            f.write(f"{text_score4}\t{caption_score4}\t{entity_scores[3]}\n")
            f.write(f"{text_score5}\t{caption_score5}\t{entity_scores[4]}\n")
        
    def compute2(self):
        pre_out = np.loadtxt("goodnews_new_pre_out.txt", delimiter="\t")
        pre_out2 = np.loadtxt("goodnews_new_pre_out2.txt", delimiter="\t")
        pre_out3 = np.loadtxt("goodnews_new_pre_out3.txt", delimiter="\t")
        pre_out4 = np.loadtxt("goodnews_new_pre_out4.txt", delimiter="\t")
        pre_out5 = np.loadtxt("goodnews_new_pre_out5.txt", delimiter="\t")
        
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
        out_file = open("goodnews_compute3.txt", "w")
        sys.stdout = out_file
        
        pre_out = np.loadtxt("goodnews_new_pre_out.txt", delimiter="\t")
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
        pre_out = np.loadtxt("/home/test/mic/codes/transfrom-and-tell-1120/goodnews_pre_out.txt", delimiter="\t")
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

    def _get_named_entities(self, article):
        # These name indices have the right end point excluded
        names = set()

        if 'context_ner' in article:
            ners = article['context_ner']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names
    
    def _get_caption_named_entities(self, caption):
        # These name indices have the right end point excluded
        names = set()

        ners = caption
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
    pre_in = open("goodnews_pre_out.txt")
    pre_out = open("goodnews_new_pre_out.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("goodnews_pre_out2.txt")
    pre_out = open("goodnews_new_pre_out2.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("goodnews_pre_out3.txt")
    pre_out = open("goodnews_new_pre_out3.txt", "w")
    for line in pre_in.readlines():
        line = line.strip() 
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("goodnews_pre_out4.txt")
    pre_out = open("goodnews_new_pre_out4.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()
    
    pre_in = open("goodnews_pre_out5.txt")
    pre_out = open("goodnews_new_pre_out5.txt", "w")
    for line in pre_in.readlines():
        line = line.strip()
        if line != "":
            pre_out.write(line + "\n")
    pre_in.close()
    pre_out.close()

if __name__ == '__main__':
    #GoodNewsStatistics().compute("train")
    #postprocess()
    GoodNewsStatistics().compute1()
    #GoodNewsStatistics().compute3()



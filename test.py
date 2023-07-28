#import torch
#import torch.nn.functional as F
import numpy as np

def test01():
    a = torch.Tensor([[1,2,3],[4,5,6]])
    mask = torch.Tensor([1, 0])
    print(a[mask.bool()])
    
def test02():
    input = torch.Tensor([[1,2],[8,4]])
    print(input)
    target = torch.LongTensor([0, 0])
    print(target)
    loss1 = F.cross_entropy(input, target, reduction="none")
    print(loss1)

def test03():
    input = torch.randn(3, 4, 5, requires_grad=True)
    print(input)
    target = torch.randint(4, (3,5), dtype=torch.int64)
    print(target)
    loss1 = F.cross_entropy(input, target, ignore_index=0, reduction="none")
    input = input.transpose(2, 1)
    input = input.contiguous().view(15, 4)
    target = target.view(-1)
    loss2 = F.cross_entropy(input, target, ignore_index=0, reduction="none")
    print(loss1, loss2)
    
def test04():
    a = {"1":"a"}
    print(list(a.items()))
    
def test05():
    import pprint
    from science_parse_api.api import parse_pdf 
    from pathlib import Path
    test_pdf_paper = Path("/home/test/2020.acl-main.2.pdf")
    host = 'http://127.0.0.1'
    port = '8080'
    output_dict = parse_pdf(host, test_pdf_paper, port=port)
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(output_dict)
    
def test06():
    import math
    print(math.log(2))
    
def test07():
    import numpy as np
    from tqdm import tqdm
    import pymongo
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.nytimes
    sample_cursor = db.articles.find({
            'split': "train",
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    sample_cursor.close()
    projection = ['_id', 'parsed_section.type', 'parsed_section.text',
                      'parsed_section.hash', 'parsed_section.parts_of_speech',
                      'parsed_section.facenet_details', 'parsed_section.named_entities',
                      'image_positions', 'headline',
                      'web_url', 'n_images_with_faces']
    count = 0
    i = 0
    for article_id in ids:
        i += 1
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}}, projection=projection)
        sections = article['parsed_section']
        image_positions = article['image_positions']
        if len(image_positions) >= 10:
            count += 1
            print(count, i)
            
def test08():
    from collections import deque
    a = deque([1,2,3,4,5,6,7,8,9,10])
    print(a.pop())
    print(a.popleft())
    print(a)
    
def test09():
    c = torch.triu(torch.ones(5,5),diagonal=1)
    print(c)

def test10():
    article_indices = torch.Tensor([1,1,2,2,2,3,3,3])
    masks = []
    for i in range(8):
        article_index = article_indices.narrow(0, i, 1)
        mask = article_indices == article_index
        masks.append(mask)
    masks = torch.stack(masks, 0)
    print(masks)
    
def test11():
    article_indices = torch.Tensor([1,1,2,2,2,3,3,3])
    a = article_indices.new(torch.ones(3,3))
    print(a)
    
def test12():
    import tell
    a = np.array([1,2,3,4,5])
    b = np.array([1,1,2])
    print(a[b])
    
def test13():
    a = torch.tensor([[1,1,0,1],
                      [1,1,0,0],
                      [0,0,0,0],
                      [1,0,0,0]]).bool()
    coh_logits1_unmasks = torch.triu(a,diagonal=1)
    b = torch.logical_or(coh_logits1_unmasks, coh_logits1_unmasks.transpose(0, 1))
    print(b)
    
def test14():
    a = torch.tensor([[1,1,0,1],
                      [1,1,0,0],
                      [0,0,0,0],
                      [1,0,0,0]])
    b = a.unique(return_inverse=True, return_counts=True)
    print(b)
    
def test15():
    import torch
    #a = torch.ones(4, 4).bool()
    a = torch.tensor([[1,1,1,0],
                      [1,1,1,0],
                      [1,1,1,0],
                      [0,0,0,1]])
    b = torch.triu(a, diagonal=1)
    c = torch.triu(a, diagonal=-1).transpose(0, 1)
    d = torch.logical_and(b, c)
    print(b)
    print(c)
    print(d)
    
def test16():
    #train: 163728 724384
    #valid: 10222 48001
    #test:  8806 41091
    import numpy as np
    from tqdm import tqdm
    import pymongo
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail
    sample_cursor = db.articles.find({
            'split': "valid", 'n_images': {"$lte": 10, "$gte": 1}
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    sample_cursor.close()
    count = 0
    image_count = 0
    i = 0
    for article_id in ids[:len(ids)//6]:
        i += 1
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
        image_positions = article['image_positions']
        if len(image_positions) <= 10:
            count += 1
            image_count += len(image_positions)
            if count  % 1000 == 0:
                print(count, image_count)
    print(count, image_count)

def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

def test17():
    import types
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from schema import And, Or, Schema, Use
    from tqdm import tqdm
    import json
    
    file1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/9_transformer_objects/serialization/generations.jsonl"
    file2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/mm1_transformer_objects/serialization/generations.jsonl"
    
    scores = {}

    with open(file1, encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            caption = obj['raw_caption']
            generation = obj['generation']
            bleu_scorer = BleuScorer(n=4)
            bleu_scorer += (generation, [caption])
            blue_score, _ = bleu_scorer.compute_score(option='closest')
            scores[obj["image_path"]] = [blue_score[3]]
            
    with open(file2, encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            caption = obj['raw_caption']
            generation = obj['generation']
            bleu_scorer = BleuScorer(n=4)
            bleu_scorer += (generation, [caption])
            blue_score, _ = bleu_scorer.compute_score(option='closest')
            s = scores[obj["image_path"]][-1]
            scores[obj["image_path"]].extend([blue_score[3], blue_score[3] - s])

    scores = list(scores.items())
    scores.sort(key=lambda x:x[-1][-1])
    files = [[x[0], x[-1][-1]] for x in scores[:200]]
    with open("/home/test/neg_images.txt", "w") as f:
        json.dump(files, f)

def test18():
    p = 0.17931664476425527
    r = 0.18271868093976773
    print(2 * p * r / (p + r)) # 0.18100167843099266
    
    p = 0.17571075996025515
    r = 0.18585085424538966
    print(2 * p * r / (p + r)) # 0.18063861624506597


def test19():
    article_indices = torch.Tensor([[1],[1],[1],[2],[2],[2],[2],[3],[3],[3]])
    btz = 10
    coh_unmasks = []
    for i in range(btz):
        article_index = article_indices.narrow(0, i, 1)
        coh_unmask = article_indices == article_index
        coh_unmasks.append(coh_unmask.squeeze(1))
    coh_unmasks = torch.stack(coh_unmasks, 0)
    b = torch.triu(coh_unmasks, diagonal=1)
    coh_unmasks = torch.logical_or(b, b.transpose(0, 1))
    print(coh_unmasks)


def test20():
    import pymongo
    from tqdm import tqdm
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.nytimes
    article_cursor = db.articles.find(projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(article_cursor)])
    article_cursor.close()
    
    count = 0
    f = open("/home/test/mic/web_url.txt", "w")
    for _, article_id in enumerate(ids):
        if count == 10000:
            break
        article = db.articles.find_one(
            {'_id': {'$eq': article_id}})
        
        if article["n_images"] <= 1:
            continue
        
        print(article["web_url"])
        f.write(article["web_url"] + "\n")
        
        if count % 1000 ==0:
            print(count)
            
        count += 1
        
    f.close()       
    

def test21():
    a = torch.ones(3,3)
    b = torch.Tensor([True, True, False])
    c = b.unsqueeze(0)
    d = b.unsqueeze(1)
    e = torch.matmul(d, c).bool()
    print(e)
    

def test22():
    a = torch.ones(3,3)
    b = torch.triu(a, diagonal=0)
    print(b)
    
    
def test23():
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_gt.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fc2.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fd1.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_f1.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fcd1_v2.jsonl"
    ps = [p1, p2, p3, p4, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                else:
                    s += 1
                c += 1
            print(s / c)        


def test23_0():
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fc2_v2_new.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fcd1_v2.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fd1_v2_new.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p4, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                else:
                    s += 0
                c += 1
            print(s / c)   
            

def test23_dm_2():
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fc2_v2_new.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fcd1_v2.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fd1_v2_new.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)   
            #print(c)

def test23_dm_0():
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fc2_v2_new.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fcd1_v2.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fd1_v2_new.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_goodnews_0():
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_fc2_v2.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_fcd1.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_fd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_0/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_goodnews_2():
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_fc2_v2.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_fcd1.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_fd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/goodnews/MMM_2/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p5]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_dailymail_0():
    print("test23_dailymail_0")
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_f1_v2.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fc2_v2_new.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fcd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_fd1_v2_new.jsonl"
    p6 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_0/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p4, p5, p6]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_dailymail_2():
    print("test23_dailymail_2")
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_f1_v2.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fc2_v2_new.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fcd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_fd1_v2_new.jsonl"
    p6 = "/home/test/mic/codes/transform-and-tell-1120/expt/dailymail/MMM_2/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p4, p5, p6]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_nytimes_0():
    print("test23_nytimes_0")
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_f1.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_fc2.2.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_fcd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_fd1_v2.2.jsonl"
    p6 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_0/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p4, p5, p6]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  
            
def test23_nytimes_2():
    print("test23_nytimes_2")
    p0 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_gt.jsonl"
    p1 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_tell.jsonl"
    p2 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_f1.jsonl"
    p3 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fc2.jsonl"
    p4 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fcd1_v2.jsonl"
    p5 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_fd1_v2.jsonl"
    p6 = "/home/test/mic/codes/transform-and-tell-1120/expt/nytimes/MMM_2/serialization/generations_coh_show.jsonl"
    ps = [p0, p1, p2, p3, p4, p5, p6]
    for p in ps:
        with open(p) as f:
            s = 0
            c = 0
            for line in f.readlines():
                line = line.strip()
                if line != "nan":
                    s += float(line)
                    
                else:
                    s += 0
                c += 1    
            print(s / c, c)  

def test24():
    coh_unmasks = torch.Tensor([[1,1,1,0],
                                [1,1,0,0],
                                [1,0,1,1],
                                [0,0,1,1]])
    b = torch.triu(coh_unmasks, diagonal=1)
    c = torch.triu(coh_unmasks, diagonal=-1)
    print(b)
    print(c)


def test25():
    from collections import deque
    a = [1,2,3,4]
    np.random.shuffle(a)
    old_list = deque(a)
    for i in old_list:
        print(i)


def test26_nytimes():
    #train: 163728 724384
    #valid: 10222 48001
    #test:  8806 41091
    import numpy as np
    from tqdm import tqdm
    import pymongo
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.nytimes
    sample_cursor = db.articles.find({}, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    sample_cursor.close()
    count = 0
    image_count = 0
    i = 0
    for article_id in ids:
        i += 1
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
        image_positions = article['image_positions']
        count += 1
        image_count += len(image_positions)
        
    print(count, image_count, image_count / count) # 445870 794217 1.7812748110435777


def test27_goodnews():
        import numpy as np
        from tqdm import tqdm
        import pymongo
        from pymongo import MongoClient
        
        client = MongoClient(host="localhost", port=27017)
        db = client.goodnews
        sample_cursor = db.splits.find({}, projection=['_id']).sort('_id', pymongo.ASCENDING)

        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        
        article_index = 0
        pre_article_id = ""
        image_count = 0
        for sample_id in ids:
            sample = db.splits.find_one({'_id': {'$eq': sample_id}})

            # Find the corresponding article
            article = db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            }, projection=['_id', 'context', 'images', 'web_url', 'caption_ner', 'context_ner', "captions_has_PGOD"])

            # Load the image
            #image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            #try:
            #    image = Image.open(image_path)
            #except (FileNotFoundError, OSError):
            #    continue
            
            if sample["article_id"] != pre_article_id:
                article_index += 1
                pre_article_id = sample["article_id"]
            
            image_count += 1
            
        
        print(article_index, image_count, image_count / article_index)


def test28_dailymail():
    #train: 163728 724384
    #valid: 10222 48001
    #test:  8806 41091
    from tqdm import tqdm
    import pymongo
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail
    sample_cursor = db.articles.find({
            'n_images': {"$lte": 10, "$gte": 1}
        }, projection=['_id']).sort('_id', pymongo.ASCENDING)
    ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
    sample_cursor.close()
    count = 0
    image_count = 0
    i = 0
    for article_id in ids:
        i += 1
        article = db.articles.find_one(
                {'_id': {'$eq': article_id}})
        image_positions = article['image_positions']
        if len(image_positions) <= 10:
            count += 1
            image_count += len(image_positions)
            if count  % 1000 == 0:
                print(count, image_count)
    print(count, image_count)
    
    
def test29():
    import spacy, random
    from pymongo import MongoClient
    nlp = spacy.load("/home/test/mic/softwares/en_core_web_lg-2.1.0")
    #doc = nlp("The windmill-shaped craft is equipped with three tractor-trailer-size soloar panels")
    doc = nlp("An Atlas V rocket with Nasa's Juno spacecraft payload sits on the launch pad at Cape Canaveral in Florida last night.")
    for ent in doc.ents:
        changed = True
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }
        print(ent)
    print()
    
        
    doc = nlp("Two elephants and the mother in South Africa.")
    for ent in doc.ents:
        changed = True
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }
        print(ent)
    
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail
    random.seed(1234)
    rs = np.random.RandomState(1234)

    entities = {}
    person_cursor = db.entities.find({'label': "PERSON"})
    entities["PERSON"] = np.array([record['text'] for record in person_cursor])
    person_cursor.close()
        
    org_cursor = db.entities.find({'label': "ORG"})
    entities["ORG"] = np.array([record['text'] for record in org_cursor])
    org_cursor.close()
        
    gpe_cursor = db.entities.find({'label': "GPE"})
    entities["GPE"] = np.array([record['text'] for record in gpe_cursor])
    gpe_cursor.close()
        
    date_cursor = db.entities.find({'label': "DATE"})
    entities["DATE"] = np.array([record['text'] for record in date_cursor])
    date_cursor.close()
    
    use_entities = ["PERSON", "ORG", "GPE", "DATE"]
    
    old_caption = "An Atlas V rocket with Nasa's Juno spacecraft payload sits on the launch pad at Cape Canaveral in Florida last night."
    doc = nlp(old_caption)
    nes = doc.ents
    for ent in doc.ents:
        changed = True
        ent_info = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': ent.text,
            'label': ent.label_,
        }
        print(ent)
    fake_caption = ""
    begin = 0
    for ne in nes:
        if ne.label_ in use_entities:
            while True:
                ch = np.random.choice(entities[ne.label_])
                print("1", ch)
                if ch != ne.text:
                    break
            fake_caption += old_caption[begin:ne.start_char]
            fake_caption += ch
            begin = ne.end_char
    fake_caption += old_caption[begin:]
    print(fake_caption)


def test30():
    from pymongo import MongoClient
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail.articles
    #items = db.find({"headline.main": {"$eq":"Celebrating America, Scoop by Frosty Scoop"}})
    items = db.find({"headline.main": {"$regex": ".*Nasa to launch probe to Jupiter just two weeks.*", "$options": "$i"}})
    #items =db.find({"source": {"$eq": "The New York Times"}})
    for item in items:
        print(item["_id"])
        

def test31():
    import hashlib
    def Hashhex(s):
      h = hashlib.sha1()
      h.update(s.encode("utf8"))
      return h.hexdigest()
    print(Hashhex("http://web.archive.org/web/20130116032220id_/http://www.dailymail.co.uk:80/sciencetech/article-2022918/Juno-takes-Nasa-launches-Juno-probe-year-mission-Jupiter-weeks-shuttle-program-ends.html"))


def test32():
    # aa1fc22a78bdaebcfb711409eef190711fe8d299
    from pymongo import MongoClient
    import json
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail.articles
    item = db.find_one({"_id": {"$eq": "aa1fc22a78bdaebcfb711409eef190711fe8d299"}})
    with open("/home/test/aa1fc22a78bdaebcfb711409eef190711fe8d299.json", "w") as f:
        json.dump(item, f)


def test33():
    import os,json
    from pymongo import MongoClient
    methods = ["9_transformer_objects", "MMM_fc2_v2_new", "MMM_fcd1_v2", "MMM_fd1_v2_new"]
    ids = {"000571afe702684d90c1d222ce70b1e1375c1016", "00119229166ae09a6ef25c0e10b101ef9eb9cca3", 
           "0013aa16650fbcfbe6edb16ac614ad174cb5d1cf", "0047e121edeb0c38199b85b824c169613a3f3dc3",
           "006bafe16458202fd775ead449eae2083841586f", "00d3c22183d528da798574bbc04cafaac69ba9d1",
           "023a4ebed179d1962e683673b90c06e68425660f", "aa1fc22a78bdaebcfb711409eef190711fe8d299",
           "0329bfa941b2f1b4ba19537bd39b94e6cacb2633", "03c51ad3996968b71843e7126ce808f917734f39",
           "0312bea2586ef3a65a1b9a3d25328d1b417e2871", "151ac17654a77f03ace5bf7d40c890feb918d244",
           "036019f373a6aaee4969ec56a8a45dd0a60d0cb0", "151a2a380771990537cd31c1882b6d25a43dc075",
           "03d642732dd32a7e493c90d46de28dbb7d31a7ff", "14db72b2f887e7c8e758a24a6a8cfdf500d812df",
           "04148ce560f474ec729e12c0a260f9678a442dc7", "145b178e2ff28cae076f46b42f0b34912246c803",
           "0523c931148e756020133531325c26351d735f66", "0cfff3751816d97e90b53a14961281d02dbcfd79", 
           "069c9ea4fc03755f1f460f8890a5c2f198646ca0", "06b6e2a8279feb2c6be3c3b8a2a7052df536a39a",
           "07478855cf4d222955dc00a8dbbb52b1cdaec873", "077af6a33b8cd89ce1334c08a6b27a0d721623e1",
           "07fe4887576224c40d2856287d28c040407885e5", "0953a85cee13aba3aebad5a7f625349e9a848679",
           "09f06b97f0a49c59c96a3319d528374b7bb57adc", "0a7803361bac620ce6539cd776bc2c71152f03f4",
           "0b00679f572fa8336ca8c4062b48a22c5243f6eb", "0ba2818cb481b9fd1933c65c26b36288e232b355"}
    
    all_captions = {}
    path = "/home/test/mic/exps"
    for method in methods:
        captions = {}
        all_captions[method] = captions
        with open(path + "/captions/" + method + "_1.jsonl") as f:
            for line in f.readlines():
                js = json.loads(line.strip())
                b = os.path.splitext(os.path.basename(js["image_path"]))[0]
                if b.split("-")[0] in ids:
                    captions[b] = js['generation'] 
        
        with open(path + "/captions/" + method + "_2.jsonl") as f:
            for line in f.readlines():
                js = json.loads(line.strip())
                b = os.path.splitext(os.path.basename(js["image_path"]))[0]
                captions[b] = js['generation']
    
    client = MongoClient(host="localhost", port=27017)
    db = client.dailymail.articles
    
    for index, article_id in enumerate(ids):
        print(index, article_id)
        article = db.find_one({"_id": {"$eq": article_id}})
        parsed_sections = article["parsed_section"]
        p1 = "{}/{:03d}".format(path, index + 1)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for method_index, method in enumerate(methods):
            with open(f"{p1}/method_{method_index + 1}.html", "w") as f:
                '''
                f.write("<html>\n")
                f.write("<head></head>\n")
                f.write("<body>\n")
                f.write("<table>\n")
                f.write("<tr>\n")
                f.write(f"<td><center>{article['headline']['main']}</center></td>\n")
                f.write("</tr>\n")
                
                for section in parsed_sections:
                    if section['type'] == "paragraph":
                        f.write("<tr>\n")
                        f.write(f"<td>{section['text']}</td>\n")
                        f.write("</tr>\n")
                    elif section['type'] == "caption":
                        f.write("<tr>\n")
                        f.write(f"<td>{section['text']}</td>\n")
                        f.write("</tr>\n")
                
                f.write("</table>\n")
                f.write("</body>\n")
                f.write("</html>\n")
                '''
                
                image_count = 0
                f.write('''<html>\n''')
                f.write('''<head><link href="../css.css" rel="stylesheet"></head>\n''')
                f.write(f'''<p class="post_title">{article['headline']['main']}</p>\n''')
                f.write('''<div class="post_body">\n''')
                for section in parsed_sections:
                    if section['type'] == "paragraph":
                        f.write(f'''<p>{section['text']}</p>\n''')
                    elif section['type'] == "caption" and image_count < 3:
                        os.system(f"cp /home/test/mic/data/dailymail/images/{section['hash']}.jpg /home/test/mic/exps/images/{section['hash']}.jpg")
                        f.write(f'''<p  class="f_center"><img src="../images/{section['hash']}.jpg" width="400" height="300"></img>\n''')
                        f.write('''<br>\n''')
                        f.write(f'''{all_captions[method][section['hash']]}\n''')
                        f.write('''</p>\n''')
                        image_count += 1
                f.write('''</div>\n''')
                f.write('''</body>\n''')
                f.write('''</html>\n''')

def test34():
    p = "D:\\research\\mic\\code\\expt\\goodnews\\MMM_fcd1\\generations.jsonl"
    with open(p) as f:
        print(f.readline())


#test23_goodnews_0()

#test23_goodnews_2()

#test23_dailymail_0()

#test23_dailymail_2()

#test23_nytimes_0()

#test23_nytimes_2()

test34()




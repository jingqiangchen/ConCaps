"""Get articles from the New York Times API.

Usage:
    process_images.py [options]

Options:
    -p --ptvsd PORT     Enable debug mode with ptvsd on PORT, e.g. 5678.
    -i --in-dir DIR   Root directory of data [default: data/goodnews/images].
    -o --out-dir DIR   Root directory of data [default: data/goodnews/images_processed].

"""
import os
from glob import glob

import ptvsd
import torchvision.transforms.functional as F
from docopt import docopt
from PIL import Image
from schema import And, Or, Schema, Use
from tqdm import tqdm

from tell.utils import setup_logger

from scripts import dailymail
from joblib import Parallel, delayed

#logger = setup_logger()

def process_data_list():
    dailymail_files = set(os.listdir(dailymail.path_stories))
    dailymail_files = set([os.path.splitext(file)[0] for file in dailymail_files])
    kb_trains = []
    kb_devs = []
    kb_tests = []
    with open(dailymail.path_corpus + "/trains-bak") as f:
        for line in f.readlines():
            kb_trains.append(line.strip())
    with open(dailymail.path_corpus + "/devs-bak") as f:
        for line in f.readlines():
            kb_devs.append(line.strip())
    with open(dailymail.path_corpus + "/tests-bak") as f:
        for line in f.readlines():
            kb_tests.append(line.strip())
    kb_trains = set(kb_trains)
    kb_devs = set(kb_devs)
    kb_tests = set(kb_tests)
    
    trains = kb_trains & dailymail_files
    devs = kb_devs & dailymail_files
    tests = kb_tests & dailymail_files
    with open(dailymail.path_corpus + "/trains", "w") as f:
        for line in trains:
            if len(os.listdir(dailymail.path_images_bak + "/" +line)) > 0:
                f.write(line + "\n")
    with open(dailymail.path_corpus + "/devs", "w") as f:
        for line in devs:
            if len(os.listdir(dailymail.path_images_bak + "/" +line)) > 0:
                f.write(line + "\n")
    with open(dailymail.path_corpus + "/tests", "w") as f:
        for line in tests:
            if len(os.listdir(dailymail.path_images_bak + "/" +line)) > 0:
                f.write(line + "\n")

# 193912 219506 0.8834018204513772
def count_captions():
    count = 0
    image_count = 0
    files = os.listdir(dailymail.path_captions) 
    for n, file in enumerate(files):
        with open(dailymail.path_captions + "/" + file) as f:
            image = len(f.readlines())
            image_count += image
            if image >= 2:
                count += 1
                if count % 1000 == 0:
                    print(count, n, image_count)
    print(count, len(files), image_count, count/len(files))



def mv_images():
    
    def v1():
        image_dir_paths = glob(f'{dailymail.path_images_bak}/*')
        for image_dir_path in tqdm(image_dir_paths):
            image_paths = glob(f"{image_dir_path}/*")
            for image_path in image_paths:
                os.system(f"mv {image_path} {image_path}.jpg")
                
    def v2():
        image_dir_paths = glob(f'{dailymail.path_images_bak}/*')
        for image_dir_path in tqdm(image_dir_paths):
            image_paths = glob(f"{image_dir_path}/*")
            for image_path in image_paths:
                os.system(f"cp {image_path} {dailymail.path_images}/{os.path.basename(image_path)}")
    
    v2()


def process_images():
    import math
    image_dir_paths = glob(f'{dailymail.path_images_bak}/*')
    def process(files):
        count = 1
        for image_dir_path in files:
            image_paths = glob(f"{image_dir_path}/*")
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                out_path = os.path.join(dailymail.path_images, image_name)
                if os.path.exists(out_path):
                    print(out_path)
                    continue
    
                try:
                    #print(f"{image_path}")
                    with Image.open(f"{image_path}") as image:
                        image = image.convert('RGB')
                        image = F.resize(image, 256, Image.ANTIALIAS)
                        image = F.center_crop(image, (224, 224))
                        image.save(out_path, image.format)
                        count += 1
                        if count % 1000 == 0:
                            print(count)
                except OSError:
                    print("error", image_path)
                    continue
    n_total = len(image_dir_paths)
    batch_size = math.ceil(n_total / 12)
    with Parallel(n_jobs=12, backend='threading') as parallel:
        parallel(delayed(process)(image_dir_paths[batch_size * i : batch_size * (i + 1)])
            for i in range(12))

def main():
    #process_data_list()
    #count_captions()
    mv_images()
    #process_images()


if __name__ == '__main__':
    main()












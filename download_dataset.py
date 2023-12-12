from datasets import load_dataset
from PIL import Image
import requests
import io
import os
import json
import hpsv2
import asyncio
from aiohttp import request
from aiomultiprocess import Pool

HPS_TARGET = 0.28

SKIP_REASON_0 = "already processed"
SKIP_REASON_1 = "label_0 equal label_1"
SKIP_REASON_2 = f"HPSv2 lower than target({HPS_TARGET})"
LOW_QUALITY_MODEL_LIST = ['stabilityai/stable-diffusion-2-1']
LOW_LIST_STR = ",".join(LOW_QUALITY_MODEL_LIST)
SKIP_REASON_3 = f"Skip Model in {LOW_LIST_STR}"

CAPTION_FOLDER = "captions"
CAPTION_EXT = ".txt"

LOG_BATCH = 50

LOG_FILE = "log.txt"

# DATASET = "yuvalkirstain/pickapic_v2_no_images"
DATASET = "F:/ImageSet/PickScore/pickapic_v2_no_images"
SPLIT = "train"

def log(content,type="info"):
    if type == "debug":
        print(content)
    if type == "info":
        # print(content)
        # open log file, create if not exist
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n")
    

# dirtory: "captions"
# file_name: "7bf0ef7c-77d5-4774-84c7-ad4c735e17f4.png"
# caption: "An anime girl, masterpiece, good line art, trending in pixiv, ,"
def save_file(dirtory,file_name,content,ext=".txt"):
    # create dirtory if not exist
    if not os.path.exists(dirtory):
        os.makedirs(dirtory)
    
    # save file by join dir and file_name
    file_path = os.path.join(dirtory,f"{file_name}{ext}")

    # write caption to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# remove file from dirtory
def remove_file(dirtory,file_name):
    # save file by join dir and file_name
    file_path = os.path.join(dirtory,f"{file_name}")
    # check file_path exist or not
    if os.path.exists(file_path):
        os.remove(file_path)

async def save_image(dirtory, file_url, file_name, downloaded_img=None):
    file_url_base = os.path.basename(file_url)
    if downloaded_img is None:
        print(f"downloading image: {file_name}")
        # Get the image content from the URL
        async with request("GET", file_url) as response:
            # Create a file-like object from the bytes
            image_file = io.BytesIO(await response.read())
            img = Image.open(image_file)
    else:
        print(f"use downloaded image: {file_name}")
        img = downloaded_img

    # Create a directory if it doesn't exist
    if not os.path.exists(dirtory):
        os.makedirs(dirtory)
    
    name, ext = os.path.splitext(file_url_base)

    # save file by join dir and file_name
    img_path = os.path.join(dirtory,f"{file_name}{ext}")
    img.save(img_path, ext[1:])

# def save_image(dirtory,file_url,file_name,downloaded_img=None):
#     file_url_base = os.path.basename(file_url)
#     if downloaded_img is None:
#         print(f"downloading image: {file_name}")
#         # Get the image content from the URL
#         r = requests.get(file_url)
#         # Create a file-like object from the bytes
#         image_file = io.BytesIO(r.content)
#         img = Image.open(image_file)
#     else:
#         print(f"use downloaded image: {file_name}")
#         img = downloaded_img

#     # Create a directory if it doesn't exist
#     if not os.path.exists(dirtory):
#         os.makedirs(dirtory)
    
#     name, ext = os.path.splitext(file_url_base)

#     # save file by join dir and file_name
#     img_path = os.path.join(dirtory,f"{file_name}{ext}")
#     img.save(img_path, ext[1:])


async def main():
    # part 1 preparation
    # create caption folder if not exist
    if not os.path.exists(CAPTION_FOLDER):
        os.makedirs(CAPTION_FOLDER)

    processed_captions = os.listdir(CAPTION_FOLDER)

    # remove the log if exist every time program started
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    download_process = {
        "processed":{},
        "skipped":{},
    }
    # Load the data from the file
    process_json = "process.json"
    if os.path.exists(process_json):
        with open(process_json, "r", encoding='utf-8') as f:
            download_process = json.load(f)
    
    # python trainscripts/imagesliders/train_lora-scale-xl.py --name 'pickscoreSliderXL' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config-xl.yaml' --folder_main 'F:/ImageSet/PickScore/images' --folders 'low, high' --scales '-1, 1'

    # part 1 load dataset
    iterable_dataset = load_dataset(DATASET,split=SPLIT, streaming=True)
    # load all from test rather than 40000
    subset = list(iterable_dataset.filter(lambda record: record["has_label"]))
    # subset = list(iterable_dataset.filter(lambda record: record["has_label"]).take(40000))

    # part 3 process
    count = 0
    indicator_folders = ["low","high"]
    # # print(take1)
    for item in subset:
        # log process
        if count % LOG_BATCH == 0:
            log(f"Log process. Item count: {count}", type="debug")
            download_process["processed_count"] = len(download_process["processed"])
            download_process["skipped_count"] = len(download_process["skipped"])
            with open(process_json, "w", encoding='utf-8') as f:
                json.dump(download_process, f)

        count = count + 1
        log(f"Count: {str(count)}", type="debug")
        # determine high image url
        high_image_url = item[f'image_0_url']
        high_image_model = item[f'model_0']
        if item['label_1'] == 1:
            high_image_url = item[f'image_1_url']
            high_image_model = item[f'model_1']

        # check best model is in LOW_QUALITY_MODEL_LIST
        if high_image_model in LOW_QUALITY_MODEL_LIST:
            log(f"Handle low quality model: {high_image_model}")
            # check process has image_id or not
            if download_process["skipped"].get(item[f'best_image_uid'],None) is None:
                # add image_id to skipped part of process
                download_process["skipped"][item[f'best_image_uid']] = SKIP_REASON_3
            
            # remove the files if stored before
            if f"{item[f'best_image_uid']}{CAPTION_EXT}" in processed_captions:
                # remove caption file
                remove_file(CAPTION_FOLDER,f"{item[f'best_image_uid']}{CAPTION_EXT}")
                log(f"{CAPTION_FOLDER}/{item[f'best_image_uid']}{CAPTION_EXT} removed")
                # remove image file
                for folder in indicator_folders:
                    remove_file(os.path.join("images",folder),f"{item[f'best_image_uid']}.png")
                    log(f"images/{folder}/{item[f'best_image_uid']}{CAPTION_EXT} removed")
            continue


        # skip processed image
        if f"{item[f'best_image_uid']}{CAPTION_EXT}" in processed_captions or item[f'best_image_uid'] in download_process["processed"].keys() or item[f'best_image_uid'] in download_process["skipped"].keys():
            if item[f'best_image_uid'] in download_process["processed"].keys():
                log(f"Count: {str(count)} Skip: {item[f'best_image_uid']} already in processed")
                continue
            if item[f'best_image_uid'] in download_process["skipped"].keys():
                log(f"Count: {str(count)} Skip: {item[f'best_image_uid']} already in skipped")
                continue
            # log missing information of processed image
            if f"{item[f'best_image_uid']}{CAPTION_EXT}" in processed_captions:
                log(f"Count: {str(count)} Skip: {item[f'best_image_uid']} already processed in processed_captions")
                download_process["processed"][item[f'best_image_uid']]={
                    'best_image_uid':str(item['best_image_uid']),
                    'caption':str(item['caption']),
                    'label_0':str(item['label_0']),
                    'label_1':str(item['label_1']),
                    'image_0_url':str(item['image_0_url']),
                    'image_1_url':str(item['image_1_url']),
                }
                # Get the image content from the URL
                r = requests.get(high_image_url)
                # Create a file-like object from the bytes
                image_file = io.BytesIO(r.content)
                img = Image.open(image_file)
                hps_score = hpsv2.score(img, item['caption'])[0]
                download_process["processed"][item[f'best_image_uid']]['hps_score'] = str(hps_score)
                log(download_process["processed"][item[f'best_image_uid']])
                continue
            # log missing information of skipped image
            if download_process["skipped"].get(item[f'best_image_uid'],None) is None:
                print(f"Count: {str(count)} Skip: {item[f'best_image_uid']} added to skipped")
                download_process["skipped"][item[f'best_image_uid']] = f"{item[f'best_image_uid']} {SKIP_REASON_0}"
            continue

        # dataset keys
        # ['are_different', 'best_image_uid', 'caption', 
        # 'created_at', 'has_label', 'image_0_uid', 'image_0_url',
        #  'image_1_uid', 'image_1_url', 'jpg_0', 'jpg_1', 'label_0', 
        # 'label_1', 'model_0', 'model_1', 'ranking_id', 'user_id', 
        # 'num_example_per_prompt', '__index_level_0__']

        # indicators = [0,1]

        # it should already filtered by dataset filter
        # skip equal images
        if item['label_0'] == item['label_1']:
            log(f"Count: {str(count)} Skip: {item[f'best_image_uid']}")
            download_process["skipped"][item[f'best_image_uid']]=SKIP_REASON_1
            continue


        # Get the image content from the URL
        # r = requests.get(high_image_url)
        # # Create a file-like object from the bytes
        # image_file = io.BytesIO(r.content)
        # img = Image.open(image_file)


        async with request("GET", high_image_url) as response:
            # Create a file-like object from the bytes
            image_file = io.BytesIO(await response.read())
            img = Image.open(image_file)

        hps_score = hpsv2.score(img, item['caption'])[0]
        # print(hps_score)

        if hps_score < HPS_TARGET:
            log(f"Count: {str(count)} Skip: {item[f'best_image_uid']} HPS: {hps_score}")
            download_process["skipped"][item[f'best_image_uid']]=SKIP_REASON_2
            continue
        # print(item[f'best_image_uid'])
        for indicator,indicator_folder in enumerate(indicator_folders):
            # print(indicator)
            image_folder = os.path.join("images",indicator_folder)
            if item[f'label_{indicator}'] == 1:
                # use the downloaded img to skip download request
                await save_image(image_folder,item[f'image_{indicator}_url'],item[f'best_image_uid'],img)
            else:
                # save image with url request
                await save_image(image_folder,item[f'image_{indicator}_url'],item[f'best_image_uid'])
        # log caption
        save_file(CAPTION_FOLDER,item[f'best_image_uid'],item['caption'])
        
        # log processed information
        download_process["processed"][item[f'best_image_uid']]={
            'best_image_uid':str(item['best_image_uid']),
            'caption':str(item['caption']),
            'label_0':str(item['label_0']),
            'label_1':str(item['label_1']),
            'image_0_url':str(item['image_0_url']),
            'image_1_url':str(item['image_1_url']),
            'hps_score':str(hps_score),
        }

if __name__ == "__main__":
    asyncio.run(main())
# download BDD private
import argparse
import json
import logging
import math
import os
import pickle
import zipfile

import numpy as np
import requests

# Check Pytorch installation
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# get image embeddings
from transformers import CLIPModel, CLIPProcessor

print(torch.__version__, torch.cuda.is_available())
# Check MMDetection installation
import mmdet

print(mmdet.__version__)
import json
import random
import subprocess

import cv2
import mmcv
from IPython.display import Image, display
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm


class BDDTeachingDataset:
    def __init__(
        self,
        data_x,
        data_y,
        ai_preds,
        image_paths,
        ai_scores,
        metadata,
        metadata_labels,
        bounding_boxes,
        captions,
        caption_embs,
        metric_y,
    ):
        self.captions = captions
        self.data_x = data_x
        self.data_y = np.array(data_y, dtype="int")
        self.ai_preds = np.array(ai_preds, dtype="int")
        self.ai_scores = ai_scores
        self.image_paths = image_paths
        self.metadata = metadata
        self.metadata_labels = metadata_labels
        self.bounding_boxes = bounding_boxes
        self.caption_embs = caption_embs
        self.metric_y = metric_y

    def get_optimal_deferral(self):
        """
        gets optimal deferral decisions computed emperically
        Return:
            opt_defer: optimal deferral decisions (binary)
        """
        opt_defer_teaching = []
        for ex in range(len(self.hum_preds)):
            score_hum = self.metric_y([self.data_y[ex]], [self.hum_preds[ex]])
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            if score_hum < score_ai:
                opt_defer_teaching.append(0)
            else:
                opt_defer_teaching.append(1)
        self.opt_defer = np.array(opt_defer_teaching)
        return np.array(opt_defer_teaching)

    def get_image(self, data_path, idx):
        # open image path
        return Image.open(data_path + self.image_paths[idx])

    def __len__(self):
        return len(self.data_y)


def download_bdd(path_data):
    if os.path.exists(os.path.join(path_data, "bdd")):
        print("bdd already downloaded")
        return
    else:
        # create foldr
        os.makedirs(os.path.join(path_data, "bdd"))

    response = requests.get(
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_det_20_labels_trainval.zip"
    )
    with open(path_data + "/bdd/bdd_labels.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(path_data + "/bdd/bdd_labels.zip", "r") as zip_ref:
        zip_ref.extractall(path_data + "/bdd")
    os.remove(path_data + "/bdd/bdd_labels.zip")
    # move from path_data/bdd/bdd100k to path_data/bdd, rename doestn work
    os.rename(path_data + "/bdd/bdd100k", path_data + "/bdd/bdd100k_labels")
    response = requests.get(
        "https://bdd-data-storage-release.s3.us-west-2.amazonaws.com/bdd100k/2021/bdd100k_images_100k.zip"
    )
    with open(path_data + "/bdd/bdd_images.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(path_data + "/bdd/bdd_images.zip", "r") as zip_ref:
        zip_ref.extractall(path_data + "/bdd")
    os.remove(path_data + "/bdd/bdd_images.zip")
    # move from path_data/bdd/bdd100k to path_data/bdd, rename doestn work
    os.rename(path_data + "/bdd/bdd100k", path_data + "/bdd/bdd100k_images")


def download_models_bdd(path_data):
    # path_data = "./models"
    logging.info("Downloading models for BDD100K")
    # check if path_data + "/faster_rcnn_r50_fpn_1x_det_bdd100k.py" exists
    if os.path.exists(
        os.path.join(path_data, "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.json")
    ):
        print("bdd models already downloaded")
        return
    response = requests.get(
        "https://github.com/SysCV/bdd100k-models/blob/main/det/configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"
    )
    with open(path_data + "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.py", "wb") as f:
        f.write(response.content)
    response = requests.get(
        "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
    )
    with open(path_data + "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.pth", "wb") as f:
        f.write(response.content)
    response = requests.get(
        "https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_1x_det_bdd100k.json"
    )
    with open(path_data + "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.json", "wb") as f:
        f.write(response.content)
    # git clone https://github.com/SysCV/bdd100k-models now
    try:
        # Use git to clone the repo
        subprocess.check_call(
            [
                "git",
                "clone",
                "git@github.com:SysCV/bdd100k-models.git",
                path_data + "/bdd/bdd100k-model",
            ]
        )
        print(f"Repository cloned successfully to {path_data + '/bdd/bdd100k-model'}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")


def prepare_bdd_dataset(path_data, device="cuda:0", ai_blur_scale=0, ai_blur_var=0):
    # download stuff
    download_bdd(path_data)
    download_models_bdd(path_data)

    with open(path_data + "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.json") as f:
        dataval_preds = json.load(f)
    with open(path_data + "/bdd/labels/det_20/det_val.json") as f:
        dataval = json.load(f)

    BDD_CLASSES = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]
    # dict of class to index
    BDD_CLASS_TO_IDX = {cls: i for i, cls in enumerate(BDD_CLASSES)}

    data_y = []
    image_paths = []
    metadata = []
    ai_preds = []
    ai_scores = []
    bounding_boxes = []
    captions = []
    metadata_labels = [
        "weather",
        "timeofday",
        "scene",
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]

    for i in tqdm(range(len(dataval))):
        is_there_traffic_light = 0
        for label in dataval[i]["labels"]:
            if label["category"] == "traffic light":
                is_there_traffic_light = 1
        predict_traffic_light = 0
        ai_score = 0
        for label in dataval_preds["frames"][i]["labels"]:
            if label["category"] == "traffic light" and label["score"] > 0.5:
                predict_traffic_light = 1
                ai_score = label["score"]
        data_y.append(is_there_traffic_light)
        image_paths.append(dataval[i]["name"])
        ai_preds.append(predict_traffic_light)
        ai_scores.append(ai_score)
        weather = dataval[i]["attributes"]["weather"]
        time_of_day = dataval[i]["attributes"]["timeofday"]
        scene = dataval[i]["attributes"]["scene"]
        objects_in_scene = np.zeros(len(BDD_CLASSES))
        for label in dataval[i]["labels"]:
            if label["category"] in BDD_CLASSES:
                objects_in_scene[BDD_CLASS_TO_IDX[label["category"]]] += 1
        metadata_i = np.concatenate(([weather, time_of_day, scene], objects_in_scene))
        metadata.append(metadata_i)
        # get bounding box for predictions of traffic light
        bounding_boxes_i = []
        for label in dataval_preds["frames"][i]["labels"]:
            if label["category"] == "traffic light":
                box2d = label["box2d"]
                box2d["score"] = label["score"]
                bounding_boxes_i.append(box2d)
        bounding_boxes_i = np.array(bounding_boxes_i)
        bounding_boxes.append(bounding_boxes_i)
        # get caption
        caption = ""
        caption += (
            scene
            + " during the "
            + time_of_day
            + " with "
            + weather
            + " weather, the scene contains "
        )
        for j in range(len(BDD_CLASSES)):
            if objects_in_scene[j] > 0 and BDD_CLASSES[j]:
                caption += str(int(objects_in_scene[j])) + " " + BDD_CLASSES[j] + "s, "
        captions.append(caption[:-2])

    # convert to numpy arrays
    data_y = np.array(data_y)
    image_paths = np.array(image_paths)
    ai_preds = np.array(ai_preds)
    ai_scores = np.array(ai_scores)
    metadata = np.array(metadata)
    bounding_boxes = np.array(bounding_boxes)
    captions = np.array(captions)

    metadata_labels = [
        "weather",
        "timeofday",
        "scene",
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]
    # alter metadata
    metadata_new = []
    for i in range(len(metadata)):
        md = []
        for j in range(len(metadata[i])):
            if j < 3:
                md.append(metadata[i][j])
            if j >= 3:
                if float(metadata[i][j]) == 0:
                    md.append("none")
                elif float(metadata[i][j]) <= 5:
                    md.append("few")
                else:
                    md.append("alot")
        metadata_new.append(md)
    metadata = np.array(metadata_new)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # need to put it on gpu if needed

    image_embeddings = []
    text_embeddings = []
    BATCH_SIZE = 15
    number_of_batches = math.ceil(len(dataval) / BATCH_SIZE)
    data_path = path_data + "/bdd/images/100k/val/"

    for batch in tqdm(range(number_of_batches)):
        images = []
        texts = []
        for i in range(BATCH_SIZE):
            if batch * BATCH_SIZE + i < len(dataval):
                if ai_blur_scale == 0 and ai_blur_var == 0:
                    images.append(
                        Image.open(data_path + image_paths[batch * BATCH_SIZE + i])
                    )
                else:
                    img = Image.open(data_path + image_paths[batch * BATCH_SIZE + i])
                    img_blurred = img.filter(
                        ImageFilter.GaussianBlur(ai_blur_var)
                    )  # this is not a perfect match, to update later
                    images.append(img_blurred)
                texts.append(captions[batch * BATCH_SIZE + i])
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        for i in range(len(texts)):
            image_embeddings.append(outputs.image_embeds[i])
            text_embeddings.append(outputs.text_embeds[i])
    # convert to numpy
    image_embeddings = torch.stack(image_embeddings).numpy()
    text_embeddings = torch.stack(text_embeddings).numpy()

    bdd_dataset = BDDTeachingDataset(
        image_embeddings,
        data_y,
        ai_preds,
        image_paths,
        ai_scores,
        metadata,
        metadata_labels,
        bounding_boxes,
        captions,
        text_embeddings,
        data_y,
    )
    with open(path_data + "/bdd/bdd_dataset.pkl", "wb") as f:
        pickle.dump(bdd_dataset, f)

    if ai_blur_scale == 0 and ai_blur_var == 0:
        return bdd_dataset
    # Choose to use a config and initialize the detector
    config = (
        path_data
        + "/bdd100k-models/det/configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py"
    )
    checkpoint = path_data + "/bdd/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
    # Set the device to be used for evaluation
    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None
    # Initialize the detector
    model = build_detector(config.model)
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    # Set the classes of models for inference

    model.CLASSES = BDD_CLASSES
    model.CLASSES = [""] * 10
    # We need to set the model's cfg for inference
    model.cfg = config
    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()
    # import time
    for i in range(1):
        img = path_data + "/bdd/images/100k/val/" + dataval[i]["name"]
        img = cv2.imread(img)
        result = inference_detector(model, img)

    truths = []
    preds = []
    scores = []
    results = []
    so_far = 0
    for indexx in tqdm(range(len(bdd_dataset.image_paths))):
        index = indexx
        img_path = path_data + "/bdd/images/100k/val/" + dataval[index]["name"]
        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img, (ai_blur_scale, ai_blur_scale), ai_blur_var)
        result = inference_detector(model, img)
        empty = result[6]
        result_new = [empty for i in range(len(result))]
        result_new[-2] = result[-2]

        is_there_traffic_light = 0
        for label in dataval[index]["labels"]:
            if label["category"] == "traffic light":
                is_there_traffic_light = 1
        green_traffic_light = 0
        for label in dataval[index]["labels"]:
            if label["category"] == "traffic light":
                if label["attributes"]["trafficLightColor"] == "G":
                    green_traffic_light = 1
        ai_says = ""
        ai_says_yes = 0
        if len(result[-2]) == 0:  # or (len(result[-2]) >0 and result[-2][0][-1]<0.5):
            ai_says = "no TL"
        else:
            ai_says = "there is " + str(result[-2][0][-1])
            ai_says_yes = 1
        score = 0
        if ai_says_yes:
            score = np.max(result[-2][:, -1])
        so_far += 1
        truths.append(is_there_traffic_light)
        results.append(result[-2])
        preds.append(ai_says_yes)
        scores.append(score)

    truths = np.array(truths)
    preds = np.array(preds)
    scores = np.array(scores)
    results = np.array(results)

    bdd_dataset = BDDTeachingDataset(
        image_embeddings,
        data_y,
        preds,
        image_paths,
        scores,
        metadata,
        metadata_labels,
        bounding_boxes,
        captions,
        text_embeddings,
        data_y,
    )
    with open(path_data + "/bdd/bdd_dataset.pkl", "wb") as f:
        pickle.dump(bdd_dataset, f)
    return bdd_dataset

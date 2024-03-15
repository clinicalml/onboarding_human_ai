import argparse
import json
import logging
import os
import random
import sys
import zipfile


# import some common libraries
import numpy as np
import requests

import cv2


sys.path.append("..")
sys.path.append("../utils")
import math
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch


from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils.utils import *


# Dataset Definition
class COCODataset:
    def __init__(
        self,
        coco_captions,
        data_x,
        data_y,
        ai_preds,
        ai_scores,
        hum_preds,
        ids,
        caption_embs,
        metric_y,
    ):
        self.coco_captions = coco_captions
        self.data_x = data_x
        self.data_y = np.array(data_y, dtype="int")
        self.ai_preds = np.array(ai_preds, dtype="int")
        self.ai_scores = ai_scores
        self.hum_preds = np.array(hum_preds, dtype="int")
        self.ids = ids
        self.caption_embs = caption_embs
        self.metric_y = metric_y
        self.get_optimal_deferral()

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

    def get_caption(self, idx):
        # just return first caption
        return self.coco_captions[idx][1][0]

    def get_image(self, idx):
        return self.coco_captions[idx][0]

    def __len__(self):
        return len(self.xs)


def download_coco(path_data):
    # download train and val images
    # check first if the images are already downloaded

    if os.path.exists(os.path.join(path_data, "/coco/train2017")):
        logging.info("train2017 already downloaded")
        return

    logging.info("downloading train2017")
    response = requests.get("http://images.cocodataset.org/zips/train2017.zip")
    with open(path_data + "/coco/train2017.zip", "wb") as f:
        f.write(response.content)
    logging.info("downloading val2017")
    response = requests.get("http://images.cocodataset.org/zips/val2017.zip")
    with open(path_data + "/coco/val2017.zip", "wb") as f:
        f.write(response.content)
    # download annotations
    response = requests.get(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    with open(path_data + "/coco/annotations_trainval2017.zip", "wb") as f:
        f.write(response.content)
    # unzip
    logging.info("unzipping")
    with zipfile.ZipFile(path_data + "/coco/train2017.zip", "r") as zip_ref:
        zip_ref.extractall(path_data)
    with zipfile.ZipFile(path_data + "/coco/val2017.zip", "r") as zip_ref:
        zip_ref.extractall(path_data)
    with zipfile.ZipFile(
        path_data + "/coco/annotations_trainval2017.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(path_data)
    # delete zip files
    os.remove(path_data + "/coco/train2017.zip")
    os.remove(path_data + "/coco/val2017.zip")
    os.remove(path_data + "/coco/annotations_trainval2017.zip")

    logging.info("Downloaded and extracted MSCOCO")


def pil_to_detectronformat(img):
    # permute first
    img = np.array(img)
    return torch.from_numpy(img)


def prepare_coco_dataset(path_data, BLURRY_DATASET=False, label_chosen=0):
    import detectron2
    from detectron2.utils.logger import setup_logger
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.modeling import build_model
    from detectron2.utils.visualizer import Visualizer
    from transformers import CLIPModel, CLIPProcessor

    if BLURRY_DATASET:
        BLUR_SIZE = 25
        BLUR_SIGMA = 17
    else:
        BLUR_SIZE = 1
        BLUR_SIGMA = 1
    # check if path_data + "/data_coco_embs_preds.pkl" exists
    if not os.path.exists(path_data + "/coco/data_coco_embs_preds.pkl"):
        setup_logger()

        download_coco(path_data)
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        model = build_model(cfg)  # returns a torch.nn.Module
        DetectionCheckpointer(model).load(
            model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        )  # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
        model.train(False)  # inference mode

        validation_coco_detection = dset.CocoDetection(
            root=path_data + "/coco/val2017",
            annFile=path_data + "/coco/annotations/instances_val2017.json",
            transform=transforms.Compose(
                [
                    transforms.GaussianBlur(BLUR_SIZE, BLUR_SIGMA),
                    transforms.PILToTensor(),
                    pil_to_detectronformat,
                ]
            ),
        )
        validation_coco_captions = dset.CocoCaptions(
            root=path_data + "/coco/val2017",
            annFile=path_data + "/coco/annotations/captions_val2017.json",
            transform=transforms.Compose(
                [
                    transforms.GaussianBlur(BLUR_SIZE, BLUR_SIGMA),
                    transforms.PILToTensor(),
                    pil_to_detectronformat,
                ]
            ),
        )

        i = 25
        im = validation_coco_detection[i][0]
        outputs = model([{"image": im}])[0]
        im = torch.permute(im, (1, 2, 0))
        im = np.array(im)
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        all_outputs = []
        all_targets = []
        all_scores = []
        BATCH_SIZE = 15
        number_of_batches = math.ceil(len(validation_coco_detection) / BATCH_SIZE)
        for batch_i in tqdm(range(number_of_batches)):
            # create batch of images
            images = []
            for i in range(
                batch_i * BATCH_SIZE,
                min((batch_i + 1) * BATCH_SIZE, len(validation_coco_detection)),
            ):
                img, target = validation_coco_detection[i]
                # append target
                categories_target = []
                for j in range(len(target)):
                    categories_target.append(target[j]["category_id"])
                all_targets.append(categories_target)
                images.append({"image": img})

            with torch.no_grad():
                outputs = model(images)
            for output in outputs:
                all_outputs.append(output["instances"].pred_classes.cpu())
                all_scores.append(output["instances"].scores.cpu())

        label_to_category_coco = {}
        with open(path_data + "/coco/coco-labels-paper.txt") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                label_to_category_coco[i + 1] = lines[i][:-1]
        print(label_to_category_coco)

        classes_rcnn = v.metadata.get("thing_classes", None)
        label_to_category_rcnn = {}
        for i in range(len(classes_rcnn)):
            label_to_category_rcnn[i] = classes_rcnn[i]

        category_to_label_rcnn = {v: k for k, v in label_to_category_rcnn.items()}
        category_to_label_coco = {v: k for k, v in label_to_category_coco.items()}

        binary_targets_categs = []
        for categ in tqdm(range(len(list(label_to_category_rcnn.keys())))):
            preds = []
            targets = []
            scores = []
            for i in range(len(all_outputs)):
                categ_coco = category_to_label_coco[label_to_category_rcnn[categ]]

                prediction = categ in all_outputs[i]
                score = 0
                if prediction:
                    for j in range(len(all_outputs[i])):
                        if all_outputs[i][j] == categ:
                            score = all_scores[i][j]
                            break

                truth = categ_coco in all_targets[i]
                scores.append(score)
                preds.append(prediction * 1.0)
                targets.append(truth * 1.0)
            binary_targets_categs.append([preds, targets, scores])

        # consider updating to openclip instead https://huggingface.co/models?library=open_clip later
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # need to put it on gpu if needed

        image_embeddings = []
        text_embeddings = []
        BATCH_SIZE = 15
        number_of_batches = math.ceil(len(validation_coco_captions) / BATCH_SIZE)

        for batch in tqdm(range(number_of_batches)):
            images = []
            texts = []
            for i in range(BATCH_SIZE):
                if batch * BATCH_SIZE + i < len(validation_coco_captions):
                    images.append(validation_coco_captions[batch * BATCH_SIZE + i][0])
                    texts.append(validation_coco_captions[batch * BATCH_SIZE + i][1][0])
            inputs = processor(
                text=texts, images=images, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)

            for i in range(len(texts)):
                image_embeddings.append(outputs.image_embeds[i])
                text_embeddings.append(outputs.text_embeds[i])
        # convert to numpy
        image_embeddings = torch.stack(image_embeddings).numpy()
        text_embeddings = torch.stack(text_embeddings).numpy()

        data = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "binary_targets_categs": binary_targets_categs,
            "label_to_category_rcnn": label_to_category_rcnn,
        }
        pickle.dump(data, open(path_data + "/coco/data_coco_embs_preds.pkl", "wb"))

    # FINAL DATA

    data_coco_storred = pickle.load(
        open(path_data + "/coco/data_coco_embs_preds.pkl", "rb")
    )
    validation_coco_detection = dset.CocoDetection(
        root=path_data + "/coco/val2017",
        annFile=path_data + "/coco/annotations/instances_val2017.json",
        transform=transforms.Compose(
            [
                transforms.GaussianBlur(BLUR_SIZE, BLUR_SIGMA),
                transforms.PILToTensor(),
                pil_to_detectronformat,
            ]
        ),
    )
    validation_coco_captions = dset.CocoCaptions(
        root=path_data + "/coco/val2017",
        annFile=path_data + "/coco/annotations/captions_val2017.json",
        transform=transforms.Compose(
            [
                transforms.GaussianBlur(BLUR_SIZE, BLUR_SIGMA),
                transforms.PILToTensor(),
                pil_to_detectronformat,
            ]
        ),
    )

    # LOAD DATA
    image_embeddings = data_coco_storred["image_embeddings"]
    text_embeddings = data_coco_storred["text_embeddings"]
    binary_targets_categs = np.array(data_coco_storred["binary_targets_categs"])
    label_to_category_rcnn = data_coco_storred["label_to_category_rcnn"]

    print(f"label chosen: {label_chosen} ({label_to_category_rcnn[label_chosen]})")
    # define human preds as being same as label 80% of the time
    hum_preds = np.zeros(len(binary_targets_categs[label_chosen][1]))
    # human is right 80% of the time
    for i in range(len(binary_targets_categs[label_chosen][1])):
        if binary_targets_categs[label_chosen][1][i] == 1:
            hum_preds[i] = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            hum_preds[i] = np.random.choice([0, 1], p=[0.8, 0.2])

    metadata_dimension = len(binary_targets_categs)
    numb_samples = len(binary_targets_categs[0][0])
    metadata = [["" for _ in range(metadata_dimension)] for __ in range(numb_samples)]
    for i in range(numb_samples):
        for j in range(metadata_dimension):
            if binary_targets_categs[j][1][i] == 1:
                metadata[i][j] = label_to_category_rcnn[j] + "present"
            else:
                metadata[i][j] = label_to_category_rcnn[j] + "absent"
    dataset = COCODataset(
        validation_coco_captions,
        image_embeddings,
        binary_targets_categs[label_chosen][1],
        binary_targets_categs[label_chosen][0],
        binary_targets_categs[label_chosen][2],
        hum_preds,
        np.arange(len(hum_preds)),
        text_embeddings,
        metric_y=loss_01,
    )
    dataset.metadata = metadata
    dataset.metadata_labels = list(label_to_category_rcnn.values())
    pickle.dump(dataset, open(path_data + "/coco/coco_dataset.pkl", "wb"))
    return dataset

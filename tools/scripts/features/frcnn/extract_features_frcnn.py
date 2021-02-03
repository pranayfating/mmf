# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import copy
import glob
import os

import numpy as np

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config


class FeatureExtractor:

    MODEL_URL = {"LXMERT": "unc-nlp/frcnn-vg-finetuned"}

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.frcnn, self.frcnn_cfg = self._build_detection_model()

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name",
            default="LXMERT",
            type=str,
            help="Model to use for detection",
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Huggingface model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Huggingface config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        if self.args.config_file:
            frcnn_cfg = Config.from_pretrained(self.args.config_file)
        else:
            frcnn_cfg = Config.from_pretrained(
                self.MODEL_URL.get(self.args.model_name, self.args.model_name)
            )
        if self.args.model_file:
            frcnn = GeneralizedRCNN.from_pretrained(
                self.args.model_file, config=frcnn_cfg
            )
        else:
            frcnn = GeneralizedRCNN.from_pretrained(
                self.MODEL_URL.get(self.args.model_name, self.args.model_name),
                config=frcnn_cfg,
            )

        return frcnn, frcnn_cfg

    def get_frcnn_features(self, image_paths):
        image_preprocess = Preprocess(self.frcnn_cfg)

        images, sizes, scales_yx = image_preprocess(image_paths)

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )

        return output_dict

    def _save_feature(self, file_name, feature):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        file_base_name = file_base_name + ".npy"
        np.save(os.path.join(self.args.output_folder, file_base_name), feature)

    def _process_features(self, features, index):
        feature_keys = [
            "obj_ids",
            "obj_probs",
            "attr_ids",
            "attr_probs",
            "boxes",
            "sizes",
            "preds_per_image",
            "roi_features",
            "normalized_boxes",
        ]
        single_features = dict()

        for key in feature_keys:
            single_features[key] = features[key][index]

        return single_features

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size], i

    def extract_features(self):
        image_dir = self.args.image_dir

        if os.path.isfile(image_dir):
            features = self.get_frcnn_features([image_dir])
            self._save_feature(image_dir, features[0])
        else:
            files = glob.glob(os.path.join(image_dir, "*.png"))
            files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
            files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

            files = {f: 1 for f in files}
            exclude = {}

            if os.path.exists(self.args.exclude_list):
                with open(self.args.exclude_list) as f:
                    lines = f.readlines()
                    for line in lines:
                        exclude[
                            line.strip("\n").split(os.path.sep)[-1].split(".")[0]
                        ] = 1

            for f in list(files.keys()):
                file_name = f.split(os.path.sep)[-1].split(".")[0]
                if file_name in exclude:
                    files.pop(f)

            files = list(files.keys())

            file_names = copy.deepcopy(files)

            start_index = self.args.start_index
            end_index = self.args.end_index
            if end_index is None:
                end_index = len(files)

            finished = 0
            total = len(files[start_index:end_index])

            for chunk, begin_idx in self._chunks(
                files[start_index:end_index], self.args.batch_size
            ):
                features = self.get_frcnn_features(chunk)
                for idx, file_name in enumerate(chunk):
                    self._save_feature(
                        file_names[begin_idx + idx],
                        self._process_features(features, idx),
                    )
                finished += len(chunk)

                if finished % 200 == 0:
                    print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()

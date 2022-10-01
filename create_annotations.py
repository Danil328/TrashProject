import json
import os.path
from argparse import ArgumentParser
from typing import List
from imagesize import get as get_size

from ok_constants.entities.datasets import (
    CocoImageInfo,
    CocoAnnotation,
    CocoCategory,
    CocoDataset,
    CocoDatasetSchema,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("predict_file", help="Path to input file")
    parser.add_argument("annotation_file", help="Path to output file")
    parser.add_argument("image_path", help="Path to image directory")
    args = parser.parse_args()
    return args


def predict_is_zbs(
    scores: List[float], low_threshold: float = 0.7, zbs_threshold: float = 0.5
) -> bool:
    low_scores = list(filter(lambda x: 0.3 < x <= low_threshold, scores))
    high_scores = list(filter(lambda x: x > low_threshold, scores))
    if len(high_scores) + len(low_scores) > 0:
        zbs_score = len(high_scores) * 1.0 / (len(high_scores) + len(low_scores))
        if zbs_score >= zbs_threshold:
            return True
    return False


def main(image_path: str, predict_file: str, annotation_file: str):
    with open(predict_file, "r") as f:
        predictions = json.load(f)

    images: List[CocoImageInfo] = []
    annotations: List[CocoAnnotation] = []
    image_id, annotation_id = 0, 0

    for folder in predictions.keys():
        for video_name in predictions[folder].keys():
            for image, prediction in predictions[folder][video_name].items():
                width, height = get_size(
                    os.path.join(image_path, folder, video_name, image)
                )
                coco_image_info = CocoImageInfo(
                    id=image_id,
                    width=width,
                    height=height,
                    file_name=os.path.join(folder, video_name, image),
                )
                flag = False
                if len(prediction["scores"]) > 0 and predict_is_zbs(
                    prediction["scores"]
                ):
                    for bbox, label, score in zip(
                        prediction["bboxes"], prediction["labels"], prediction["scores"]
                    ):
                        if score >= 0.35:
                            coco_annotation = CocoAnnotation(
                                id=annotation_id,
                                image_id=image_id,
                                category_id=label,
                                segmentation=[
                                    [
                                        bbox[0],
                                        bbox[1],
                                        bbox[0],
                                        (bbox[1] + bbox[3]),
                                        (bbox[0] + bbox[2]),
                                        (bbox[1] + bbox[3]),
                                        (bbox[0] + bbox[2]),
                                        bbox[1],
                                    ]
                                ],
                                bbox=bbox,
                                area=int(bbox[2] * bbox[3]),
                            )
                            annotation_id += 1
                            annotations.append(coco_annotation)
                            flag = True
                if flag:
                    images.append(coco_image_info)
                    image_id += 1

    category_mapping = dict(bottle_blue=0, bottle_white=1, bottle_brown=2, plastic=3)
    categories = []
    for key, value in category_mapping.items():
        category = CocoCategory(name=key, id=value)
        categories.append(category)
    coco_dataset = CocoDataset(
        images=images, annotations=annotations, categories=categories
    )
    print(f"Num annotations - {len(annotations)}")
    print(f"Num images - {len(images)}")
    with open(annotation_file, "w") as f:
        json.dump(CocoDatasetSchema().dump(coco_dataset), f, sort_keys=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.image_path, args.predict_file, args.annotation_file)

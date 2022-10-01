import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from typing import List, Dict, Tuple

from ok_constants.entities.datasets import (
    CocoCategory,
    CocoDataset,
    CocoAnnotation,
    CocoImageInfo,
    CocoDatasetSchema,
)
from tqdm import tqdm


def get_image_size(root: ET.Element) -> Tuple[int, int, int]:
    r = {}
    for child in root.iter("size"):
        for param in child:
            r[param.tag] = param.text
    return int(r["width"]), int(r["height"]), int(r["depth"])


def decode_bbox(root: ET.Element) -> List[float]:
    xmin = int(root.find("xmin").text)
    ymin = int(root.find("ymin").text)
    xmax = int(root.find("xmax").text)
    ymax = int(root.find("ymax").text)
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def decode_polygons(root: ET.Element) -> List[List[float]]:
    polygons = []
    polygon = root.find("polygon")
    for point in polygon.iter("point"):
        for value in point.iter("value"):
            polygons.append(int(value.text))
    return [polygons]


def get_annotations(
    root: ET.Element,
    category_mapping: Dict[str, int],
    image_id: int,
    annotation_id: int,
) -> Tuple[List[CocoAnnotation], int]:
    annotations = []
    for obj in root.iter("object"):
        _id = obj.find("_id").text
        category = obj.find("name").text
        bbox = decode_bbox(obj.find("bndbox"))
        generate_type = obj.find("generate_type").text
        file_id = obj.find("file_id").text
        polygons = decode_polygons(obj.find("segment_polygons"))

        coco_annotation = CocoAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=category_mapping[category],
            segmentation=polygons,
            bbox=bbox,
            area=int(bbox[2] * bbox[3]),
        )
        annotations.append(coco_annotation)
        annotation_id += 1

    return annotations, annotation_id


def main(data_path: str, output_path: str):
    images: List[CocoImageInfo] = []
    annotations: List[CocoAnnotation] = []
    category_mapping = dict(bottle_blue=0, bottle_white=1, bottle_brown=2, plastic=3)

    image_id, annotation_id = 0, 0
    folders = os.listdir(data_path)
    for folder in tqdm(folders, desc="Create COCO annotaion"):
        if folder.startswith("GOOOD") and os.path.isdir(
            os.path.join(data_path, folder)
        ):
            for xml in glob.glob(os.path.join(data_path, folder, "*.xml")):
                root = ET.parse(xml).getroot()
                width, height, channels = get_image_size(root)
                coco_image_info = CocoImageInfo(
                    id=image_id,
                    width=width,
                    height=height,
                    file_name=xml.split("/")[-1].replace(".xml", ".png"),
                )
                images.append(coco_image_info)

                annotation, annotation_id = get_annotations(
                    root,
                    category_mapping=category_mapping,
                    image_id=image_id,
                    annotation_id=annotation_id,
                )
                annotations.extend(annotation)
                image_id += 1
    categories = []
    for key, value in category_mapping.items():
        category = CocoCategory(name=key, id=value)
        categories.append(category)
    coco_dataset = CocoDataset(
        images=images, annotations=annotations, categories=categories
    )
    with open(output_path, "w") as f:
        json.dump(CocoDatasetSchema().dump(coco_dataset), f, sort_keys=True)
    return coco_dataset


def move_images(old_images_path: str, new_images_path: str):
    os.makedirs(new_images_path, exist_ok=True)
    for folder in tqdm(os.listdir(old_images_path), desc="Move images"):
        if folder.startswith("GOOOD") and os.path.isdir(
            os.path.join(old_images_path, folder)
        ):
            for image in os.listdir(os.path.join(old_images_path, folder)):
                if image.endswith(".png") or image.endswith(".jpg"):
                    shutil.move(
                        os.path.join(old_images_path, folder, image),
                        os.path.join(new_images_path, image),
                    )


parser = ArgumentParser(description="Read images from disk, crop and upload to YT.")
parser.add_argument(
    "--data-path", type=str, default="data", help="Path to image directory"
)
parser.add_argument(
    "--output-path",
    type=str,
    default="data/coco_annotation.json",
    help="Path to coco annotation",
)
parser.add_argument(
    "--images-path", type=str, default="data/images", help="Path to image directory"
)
if __name__ == "__main__":
    args = parser.parse_args()
    main(args.data_path, args.output_path)
    move_images(args.data_path, args.images_path)

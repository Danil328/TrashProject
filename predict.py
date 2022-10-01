import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("images", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--out-file", default=None, help="Path to output file")
    parser.add_argument("--batch-size", default=32, help="Batch size")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=0.2, help="bbox score threshold"
    )
    args = parser.parse_args()
    return args


def main(
    image_path: str,
    prediction_path: str,
    config_path: str,
    checkpoint_path: str,
    threshold: float,
    batch_size: int,
    device: str,
):
    predictions = defaultdict(dict)
    model = init_detector(config_path, checkpoint_path, device=device)
    for folder in list(filter(lambda x: x.startswith("2022"), os.listdir(image_path))):
        predictions[folder] = defaultdict(dict)
        for video_name in list(
            filter(
                lambda x: not x.startswith("."),
                os.listdir(os.path.join(image_path, folder)),
            )
        ):
            images = list(
                map(
                    lambda x: os.path.join(image_path, folder, video_name, x),
                    os.listdir(os.path.join(image_path, folder, video_name)),
                )
            )
            for i in tqdm(
                range(0, len(images), batch_size),
                desc=f"Predict folder - {folder}/{video_name}",
            ):
                batch = images[i : i + batch_size]
                bbox_results = inference_detector(model, batch)
                for image, bbox_result in zip(batch, bbox_results):
                    bboxes = np.vstack(bbox_result)
                    scores = bboxes[:, -1]
                    bboxes = bboxes[:, :4]
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
                    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
                    idx = np.where(scores >= threshold)
                    labels = [
                        np.full(bbox.shape[0], j, dtype=np.int32)
                        for j, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)
                    image_predictions = dict(
                        bboxes=bboxes[idx].tolist(),
                        scores=scores[idx].tolist(),
                        labels=labels[idx].tolist(),
                    )
                    predictions[folder][video_name][
                        image.split("/")[-1]
                    ] = image_predictions
    with open(prediction_path, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.images,
        args.out_file,
        args.config,
        args.checkpoint,
        args.score_thr,
        args.batch_size,
        args.device,
    )

import os
from argparse import ArgumentParser

from ok_datasets.preprocessing import IdsVideoClassificationPreprocessor
from tqdm import tqdm


def main(video_path: str, frames_path: str, num_frames: int):
    preprocessor = IdsVideoClassificationPreprocessor()
    for folder in os.listdir(video_path):
        os.makedirs(os.path.join(frames_path, folder), exist_ok=True)
        for video in tqdm(
            list(
                filter(
                    lambda x: x.endswith(".avi"),
                    os.listdir(os.path.join(video_path, folder)),
                )
            ),
            desc="Extract frames",
        ):
            preprocessor.split_video(
                video,
                os.path.join(video_path, folder),
                os.path.join(frames_path, folder),
                num_frames=num_frames,
                target_shape=None,
                file_extension=".jpg",
                use_key_frames=False,
            )


parser = ArgumentParser(description="Read images from disk, crop and upload to YT.")
parser.add_argument(
    "--video-path", type=str, default="data", help="Path to video directory"
)
parser.add_argument(
    "--frames-path", type=str, default="data", help="Path to frames directory"
)
parser.add_argument(
    "--num-frames", type=int, default=500, help="Number of frames per video"
)
if __name__ == "__main__":
    args = parser.parse_args()
    main(args.video_path, args.frames_path, args.num_frames)

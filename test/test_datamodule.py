import torch
import hydra
import sys
import os

sys.path.append("src")
from hydra.utils import instantiate
import socket
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def test_dm():
    hydra.initialize_config_dir("/root/projects/sign_language_multitask/configs")
    cfg = hydra.compose("run/train/resnet_efficient_causal.yaml")
    # cfg = hydra.compose('run/train/dual')
    print(socket.gethostname())
    # cfg.datamodule.num_workers = 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        datamodule = instantiate(cfg.datamodule, thread_pool=executor, num_workers=12)
        # loader = datamodule.train_dataloader()
        # for batch in tqdm(loader):
        #     pass
        dataset = datamodule.train_set
        dataset.transform = None
        import re

        for id in dataset.annotation.id:
            # print(id)
            if re.match("13April_2011_Wednesday_tagesschau_default", id):
                print(id)
                # break

        for i in tqdm(range(len(dataset))):
            data = dataset[i]

        # Draw all frames using matplotlib
        video_frames = data["video"]
        output_dir = "outputs/frames"
        os.makedirs(output_dir, exist_ok=True)

        for i, frame in enumerate(video_frames):
            plt.imshow(frame)
            plt.title(f"Frame {i}")
            plt.pause(0.001)  # Pause to allow the plot to update
            input("Press Enter to display the next frame...")


if __name__ == "__main__":
    test_dm()

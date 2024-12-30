import os

BASE = "/projects/perception/datasets/4dunderstanding/data/scannet"

scans = sorted(os.listdir(BASE))

for scan in scans:

    old_dir = os.path.join(BASE, scan, "color_90")
    new_dir = os.path.join(BASE, scan, "rgb_aligned")

    os.rename(old_dir, new_dir)

    old_dir = os.path.join(BASE, scan, "depth_90")
    new_dir = os.path.join(BASE, scan, "depth_aligned")

    os.rename(old_dir, new_dir)
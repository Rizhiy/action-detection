# Owner: Artem Vasenin (a.vasenin@ntechlab.com)

import re
import sys
from pathlib import Path
from typing import Tuple

import cv2
import mxnet as mx
import numpy as np
from PIL import Image

base_dir = Path('/media/ActionDetection/data/THUMOS/frames')
dest_dir = Path('/media/ActionDetection/data/THUMOS/compressed_frames')


def convert_folder(folder: Path, dest_dir: Path):
    result_path = dest_dir / folder.stem
    print('Compressing: {:70s} -> {:70s}'.format(
        str(folder), str(result_path.with_suffix('.rec'))), end='   ')
    sys.stdout.flush()
    with open(result_path.with_suffix('.img_idx'), 'w') as f:
        record = mx.recordio.MXIndexedRecordIO(str(result_path.with_suffix('.idx')),
                                               str(result_path.with_suffix('.rec')), 'w')
        for idx, img in enumerate(sorted(sorted(folder.iterdir()),
                                         key=lambda x: int(re.findall('\d+', str(x))[-1]))):
            f.write(img.stem + '\n')
            record.write_idx(idx, (folder / img).read_bytes())
        record.close()
    print('=Done=')


def convert_frames(folder: Path):
    for video in sorted(folder.iterdir()):
        dest_dir.mkdir(exist_ok=True)
        convert_folder(video, dest_dir)


class BinaryDataset:
    def __init__(self, base_folder: Path, cache_size: int = 8):
        self.base_folder = base_folder
        self.cache = {}
        self.cache_size = cache_size

    def __getitem__(self, x: Tuple[Path, str]) -> Image.Image:
        folder, img = x
        if folder not in self.cache:
            result_path = (self.base_folder / folder)
            if len(self.cache) >= self.cache_size:
                self.cache.popitem()
            with open(result_path.with_suffix('.img_idx'), 'r') as f:
                img_idx = [x.strip() for x in f.readlines()]
            self.cache[folder] = {
                'record':  mx.recordio.MXIndexedRecordIO(str(result_path.with_suffix('.idx')),
                                                         str(result_path.with_suffix('.rec')), 'r'),
                'img_idx': {x: idx for idx, x in enumerate(img_idx)}}
        index = self.cache[folder]['img_idx'][img]
        img_bytes = self.cache[folder]['record'].read_idx(index)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return Image.fromarray(cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1])


if __name__ == '__main__':
    convert_frames(base_dir)

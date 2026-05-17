import json
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


"""
QuickDraw absolute strokes
-> relative stroke-3: [dx, dy, pen_up]
-> normalized stroke-5: [dx, dy, pen_down, pen_up, end]
-> train with input sequence shifted against target sequence
"""

class SketchDataset(Dataset):
    def __init__(self, dataset: list[np.ndarray], max_seq_len: int, scale: Optional[float] = None):
        data = []

        for seq in dataset:           # seq: [[dx, dy, pen_up], ...] : (sequence_length, 3)
            if 10 < len(seq) <= max_seq_len:
                seq = np.clip(seq, -1000, 1000).astype(np.float32)
                data.append(seq)

        if not data:
            raise ValueError("No sketches left after filtering. Try increasing --max-seq-len or --max-samples.")

        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = float(scale) if scale and scale > 0 else 1.0

        longest_seq_len = max(len(seq) for seq in data)

        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float32)
        self.mask = torch.zeros(len(data), longest_seq_len + 1, dtype=torch.float32)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)

            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / self.scale
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]  # pen_up -> pen_down
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]      # pen_up
            self.data[i, len_seq + 1:, 4] = 1               # end token: [0, 0, 0, 0, 1]
            self.mask[i, :len_seq + 1] = 1                  # stat + seq = valid(1) 

        self.data[:, 0, 2] = 1    # start token: [0, 0, 1, 0, 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.mask[idx]


def drawing_to_strokes(drawing: Any) -> np.ndarray:
    """Convert a QuickDraw drawing into SketchRNN stroke-3: dx, dy, pen_up."""
    strokes = []
    prev_x, prev_y = 0.0, 0.0

    for stroke in drawing:
        if len(stroke) != 2:
            continue

        xs, ys = stroke
        for point_idx, (x, y) in enumerate(zip(xs, ys)):
            pen_up = 1.0 if point_idx == len(xs) - 1 else 0.0
            strokes.append([float(x) - prev_x, float(y) - prev_y, pen_up])
            prev_x, prev_y = float(x), float(y)

    return np.asarray(strokes, dtype=np.float32)


def load_quickdraw_ndjson(path: str, max_items: Optional[int] = None, recognized_only: bool = True) -> list[np.ndarray]:
    sketches = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if recognized_only and not item.get("recognized", False):
                continue

            strokes = drawing_to_strokes(item["drawing"])
            if len(strokes) > 0:
                sketches.append(strokes)

            if max_items is not None and len(sketches) >= max_items:
                break

    return sketches

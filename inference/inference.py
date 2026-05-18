import argparse
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import DecoderLSTM, sample_sketch


def parse_args():
    parser = argparse.ArgumentParser(description="Sample sketches from a trained SketchRNN checkpoint.")
    parser.add_argument("--checkpoint", default=r"C:\Users\Aryan\Documents\SketchRNN\runs\bicycle\checkpoint_epoch_030.pt")
    parser.add_argument("--out", default="samples/bicycle_sample")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_decoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint["args"]
    decoder = DecoderLSTM(
        train_args["d_z"],
        train_args["dec_hidden_size"],
        train_args["n_distributions"],
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    return decoder, train_args, checkpoint["scale"]





def strokes_to_svg(strokes, output_path, padding=16):
    x, y = 0.0, 0.0
    paths = []
    current = []

    for dx, dy, pen_up in strokes:
        x += float(dx)
        y += float(dy)
        current.append((x, y))
        if pen_up > 0.5 and current:
            paths.append(current)
            current = []

    if current:
        paths.append(current)

    points = [point for path in paths for point in path]
    if not points:
        points = [(0.0, 0.0)]

    min_x = min(p[0] for p in points) - padding
    min_y = min(p[1] for p in points) - padding
    max_x = max(p[0] for p in points) + padding
    max_y = max(p[1] for p in points) + padding
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"{min_x:.2f} {min_y:.2f} {width:.2f} {height:.2f}",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(svg, "rect", {"x": f"{min_x:.2f}", "y": f"{min_y:.2f}", "width": f"{width:.2f}", "height": f"{height:.2f}", "fill": "white"})

    for path in paths:
        if len(path) == 1:
            ET.SubElement(svg, "circle", {"cx": f"{path[0][0]:.2f}", "cy": f"{path[0][1]:.2f}", "r": "1.5", "fill": "black"})
            continue

        commands = [f"M {path[0][0]:.2f} {path[0][1]:.2f}"]
        commands.extend([f"L {px:.2f} {py:.2f}" for px, py in path[1:]])
        ET.SubElement(
            svg,
            "path",
            {
                "d": " ".join(commands),
                "fill": "none",
                "stroke": "black",
                "stroke-width": "2.5",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )

    ET.ElementTree(svg).write(output_path, encoding="utf-8", xml_declaration=True)


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device(args.device)
    decoder, train_args, scale = load_decoder(args.checkpoint, device)
    strokes = sample_sketch(decoder, train_args["d_z"], scale, args.max_len, args.temperature, device)

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_base.with_suffix(".npy"), strokes)
    strokes_to_svg(strokes, out_base.with_suffix(".svg"))
    print(f"Wrote {len(strokes)} points to {out_base.with_suffix('.npy')} and {out_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

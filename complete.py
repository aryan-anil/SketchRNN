import argparse
from pathlib import Path
import random
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
import torch.nn.functional as F

from inference import strokes_to_svg
from model import DecoderLSTM, EncoderLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Draw a partial sketch and let SketchRNN complete it.")
    parser.add_argument("--checkpoint", default=r"C:\Users\Aryan\Documents\SketchRNN\runs\bicycle\checkpoint_epoch_030.pt")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--min-continue-steps", type=int, default=12)
    parser.add_argument("--canvas-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint["args"]

    encoder = EncoderLSTM(train_args["d_z"], train_args["enc_hidden_size"]).to(device)
    decoder = DecoderLSTM(
        train_args["d_z"],
        train_args["dec_hidden_size"],
        train_args["n_distributions"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, train_args, checkpoint["scale"]


def simplify_path(path, min_distance=3.0):
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    last_x, last_y = path[0]

    for x, y in path[1:-1]:
        if ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 >= min_distance:
            simplified.append((x, y))
            last_x, last_y = x, y

    if simplified[-1] != path[-1]:
        simplified.append(path[-1])

    return simplified


def points_to_strokes(paths):
    strokes = []
    prev_x, prev_y = 0.0, 0.0

    for path in paths:
        if not path:
            continue

        for point_idx, (x, y) in enumerate(path):
            pen_up = 1.0 if point_idx == len(path) - 1 else 0.0
            strokes.append([float(x) - prev_x, float(y) - prev_y, pen_up])
            prev_x, prev_y = float(x), float(y)

    return np.asarray(strokes, dtype=np.float32)


def normalize_paths_for_model(paths, target_size=180.0, model_canvas_size=256.0):
    simplified_paths = [simplify_path(path) for path in paths if path]
    points = [point for path in simplified_paths for point in path]
    if not points:
        return simplified_paths, 1.0

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    longest_side = max(width, height)
    canvas_to_model = target_size / longest_side

    offset_x = (model_canvas_size - width * canvas_to_model) / 2.0
    offset_y = (model_canvas_size - height * canvas_to_model) / 2.0

    model_paths = []
    for path in simplified_paths:
        model_path = []
        for x, y in path:
            model_x = (x - min_x) * canvas_to_model + offset_x
            model_y = (y - min_y) * canvas_to_model + offset_y
            model_path.append((model_x, model_y))
        model_paths.append(model_path)

    return model_paths, canvas_to_model


def stroke3_to_stroke5(strokes, scale, device, force_continue=False):
    data = torch.zeros(1, len(strokes) + 1, 5, dtype=torch.float32, device=device)
    data[0, 0, 2] = 1.0

    if len(strokes) > 0:
        stroke_tensor = torch.from_numpy(strokes).to(device)
        data[0, 1:, :2] = stroke_tensor[:, :2] / scale
        data[0, 1:, 2] = 1.0 - stroke_tensor[:, 2]
        data[0, 1:, 3] = stroke_tensor[:, 2]
        if force_continue:
            data[0, -1, 2] = 1.0
            data[0, -1, 3] = 0.0

    return data.transpose(0, 1)


def sample_step(dist, q_log_probs, temperature):
    dist.set_temperature(temperature)
    cat_dist, multi_dist = dist.get_distribution()
    mixture_idx = cat_dist.sample()
    xy_samples = multi_dist.sample()
    xy = xy_samples.gather(-2, mixture_idx[..., None, None].expand(*mixture_idx.shape, 1, 2)).squeeze(-2)

    pen_logits = q_log_probs / temperature
    pen = torch.distributions.Categorical(probs=F.softmax(pen_logits, dim=-1)).sample()
    return xy.squeeze(0).squeeze(0), int(pen.item())


def complete_strokes(
    encoder,
    decoder,
    model_strokes,
    scale,
    max_len,
    temperature,
    device,
    canvas_to_model=1.0,
    min_continue_steps=12,
):
    prefix = stroke3_to_stroke5(model_strokes, scale, device, force_continue=False)

    with torch.no_grad():
        # Use the latent from the prefix
        z, mu, log_var = encoder(prefix)
        # For more stable completion, you can use mu instead of z:
        # z = mu

        # Run the whole prefix through the decoder
        state = None
        dist = None
        q_log_probs = None
        for t in range(prefix.shape[0]):
            dist, q_log_probs, state = decoder(prefix[t:t+1], z, state)

        generated = []

        # The last `dist` is already the distribution for the next step
        for step_idx in range(max_len):
            xy, pen = sample_step(dist, q_log_probs, temperature)

            if pen == 2 and step_idx >= min_continue_steps:
                break

            dx = float(xy[0].item() * scale / canvas_to_model)
            dy = float(xy[1].item() * scale / canvas_to_model)
            pen_up = 1.0 if pen == 1 else 0.0
            generated.append([dx, dy, pen_up])

            prev = torch.tensor(
                [[xy[0].item(), xy[1].item(), 1.0 - pen_up, pen_up, 0.0]],
                dtype=torch.float32,
                device=device
            ).view(1, 1, 5)

            dist, q_log_probs, state = decoder(prev, z, state)

    if not generated:
        return np.empty((0, 3), dtype=np.float32)

    return np.asarray(generated, dtype=np.float32)

class CompletionApp:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.encoder, self.decoder, self.train_args, self.scale = load_models(args.checkpoint, self.device)

        self.paths = []
        self.current_path = []
        self.generated_strokes = np.empty((0, 3), dtype=np.float32)
        self.last_point = None

        self.root = tk.Tk()
        self.root.title("SketchRNN Completion")

        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(toolbar, text="Complete", command=self.complete).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(toolbar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(toolbar, text="Save SVG", command=self.save_svg).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(toolbar, text="Temperature").pack(side=tk.LEFT, padx=(16, 4))
        self.temperature = tk.DoubleVar(value=args.temperature)
        tk.Scale(toolbar, variable=self.temperature, from_=0.1, to=1.2, resolution=0.05, orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT)

        self.status = tk.StringVar(value=f"Loaded {args.checkpoint}")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8)

        self.canvas = tk.Canvas(self.root, width=args.canvas_size, height=args.canvas_size, bg="white", cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.extend_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

    def run(self):
        self.root.mainloop()

    def start_stroke(self, event):
        self.current_path = [(event.x, event.y)]
        self.last_point = (event.x, event.y)

    def extend_stroke(self, event):
        if self.last_point is None:
            return

        x0, y0 = self.last_point
        self.canvas.create_line(x0, y0, event.x, event.y, fill="black", width=3, capstyle=tk.ROUND, smooth=True)
        self.current_path.append((event.x, event.y))
        self.last_point = (event.x, event.y)

    def end_stroke(self, event):
        if self.current_path:
            if len(self.current_path) == 1:
                x, y = self.current_path[0]
                self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="black", outline="black")
            self.paths.append(self.current_path)

        self.current_path = []
        self.last_point = None

    def complete(self):
        if not self.paths:
            messagebox.showinfo("SketchRNN Completion", "Draw a few strokes first.")
            return

        user_strokes = points_to_strokes(self.paths)
        if len(user_strokes) < 2:
            messagebox.showinfo("SketchRNN Completion", "Draw a little more before completing.")
            return

        self.status.set("Completing...")
        self.root.update_idletasks()

        try:
            model_paths, canvas_to_model = normalize_paths_for_model(self.paths)
            model_strokes = points_to_strokes(model_paths)
            self.canvas.delete("generated")
            self.generated_strokes = complete_strokes(
                self.encoder,
                self.decoder,
                model_strokes,
                self.scale,
                self.args.max_len,
                self.temperature.get(),
                self.device,
                canvas_to_model=canvas_to_model,
                min_continue_steps=self.args.min_continue_steps,
            )
            self.draw_generated(user_strokes, self.generated_strokes)
            self.status.set(f"Added {len(self.generated_strokes)} generated points.")
        except Exception as exc:
            messagebox.showerror("SketchRNN Completion", str(exc))
            self.status.set("Completion failed.")

    def draw_generated(self, user_strokes, generated_strokes):
        if len(generated_strokes) == 0:
            return

        x, y = 0.0, 0.0
        for dx, dy, _ in user_strokes:
            x += float(dx)
            y += float(dy)

        pen_is_down = True
        for dx, dy, pen_up in generated_strokes:
            next_x = x + float(dx)
            next_y = y + float(dy)
            if pen_is_down:
                self.canvas.create_line(
                    x,
                    y,
                    next_x,
                    next_y,
                    fill="#2563eb",
                    width=3,
                    capstyle=tk.ROUND,
                    smooth=True,
                    tags=("generated",),
                )
            pen_is_down = pen_up < 0.5
            x, y = next_x, next_y

    def clear(self):
        self.canvas.delete("all")
        self.paths = []
        self.current_path = []
        self.generated_strokes = np.empty((0, 3), dtype=np.float32)
        self.last_point = None
        self.status.set("Canvas cleared.")

    def save_svg(self):
        if not self.paths:
            messagebox.showinfo("SketchRNN Completion", "Nothing to save yet.")
            return

        user_strokes = points_to_strokes(self.paths)
        strokes = np.concatenate([user_strokes, self.generated_strokes], axis=0)
        output_path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg")],
            initialfile="completed_sketch.svg",
        )
        if not output_path:
            return

        strokes_to_svg(strokes, Path(output_path))
        self.status.set(f"Saved {output_path}")


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    app = CompletionApp(args)
    app.run()


if __name__ == "__main__":
    main()

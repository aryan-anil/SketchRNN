# SketchRNN

Implementation of SketchRNN, inspired from:

- https://nn.labml.ai/sketch_rnn/index.html



## Project Structure

```text
SketchRNN/
  data/              # QuickDraw .ndjson files, ignored by git
  inference/         # Sampling and sketch-completion scripts
  model/             # Encoder, decoder, mixture distribution, sampling code
  runs/              # Training checkpoints, ignored by git
  samples/           # Generated .npy/.svg outputs, ignored by git
  train/             # Dataset loader and training loop
```



Original paper:

```bibtex
@article{ha2017neural,
  title={A neural representation of sketch drawings},
  author={Ha, David and Eck, Douglas},
  journal={arXiv preprint arXiv:1704.03477},
  year={2017}
}
```

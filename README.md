# SketchRNN

Implementation of SketchRNN, inspired from:

- https://nn.labml.ai/sketch_rnn/index.html



## Project Structure

```text
SketchRNN/
  data/              # QuickDraw .ndjson files, download and place data here
  inference/         # Sketch sampling scripts
  model/             # Encoder, decoder, mixture distribution, sampling code
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

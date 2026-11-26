"""Export an LSTM checkpoint for the offline web demo in web/.

Writes web/model.onnx (input: int64 ids [1, T] -> output: next-word logits [V]) and web/vocab.json (itos).
Usage: python export_web.py [checkpoint]   (default: models/best.pt)
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/best.pt"
OUT = "web"


class LSTMLanguageModel(nn.Module):  # same state_dict keys as script.ipynb / streamlit_app.py
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, tie_weights, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim) if tie_weights and hidden_dim != embed_dim else None
        self.decoder = nn.Linear(embed_dim if self.proj is not None else hidden_dim, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        out = out[0, -1]  # only the last position is needed for a suggestion
        if self.proj is not None:
            out = self.proj(out)
        return self.decoder(out)


ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
c, itos = ckpt["config"], ckpt["vocab"]
model = LSTMLanguageModel(len(itos), c["embed_dim"], c["hidden_dim"], c["num_layers"],
                          c["tie_weights"], itos.index("<pad>"))
model.load_state_dict(ckpt["model"])  # tied decoder.weight is stored too, so plain loading works
model.eval()

os.makedirs(OUT, exist_ok=True)
onnx_path = os.path.join(OUT, "model.onnx")
dummy = torch.tensor([[itos.index("<eos>"), 4, 5, 6]])
torch.onnx.export(model, (dummy,), onnx_path, input_names=["ids"], output_names=["logits"],
                  dynamic_axes={"ids": {1: "T"}}, opset_version=17, dynamo=False)
with open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(itos, f, ensure_ascii=False)

# self-check: ONNX output matches PyTorch for a few prefix lengths
sess = ort.InferenceSession(onnx_path)
for T in (1, 5, 40):
    ids = torch.randint(4, len(itos), (1, T))
    ids[0, 0] = itos.index("<eos>")
    ref = model(ids).detach().numpy()
    got = sess.run(None, {"ids": ids.numpy()})[0]
    assert np.allclose(ref, got, atol=1e-4), f"mismatch at T={T}: {np.abs(ref - got).max()}"
print(f"ok: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB), vocab {len(itos)} words")

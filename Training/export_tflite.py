import os
import subprocess
import torch
import torch.nn as nn

# MODEL
class TinyRULCNN(nn.Module):
    def __init__(self, input_features=11):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_features, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = torch.mean(x, dim=2)
        x = self.regressor(x)
        return x.flatten()

PTH_PATH = "RUL_TINYML.pth"

ONNX_PATH = "tiny_rul.onnx"

TF_DIR = "TinyRUL_tf"

TFLITE_PATH = os.path.join(
    TF_DIR,
    "tiny_rul_float32.tflite"
)

HEADER_PATH = "model_finalVersion.h"

# LOAD MODEL
print("Loading checkpoint...")

checkpoint = torch.load(
    PTH_PATH,
    map_location="cpu",
    weights_only=False
)

model = TinyRULCNN(
    checkpoint["input_features"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
# EXPORT ONNX
print("Exporting ONNX...")

dummy_input = torch.randn(
    1,
    10,
    11
)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=16,
    input_names=["input"],
    output_names=["rul_prediction"]
)

print("ONNX saved.")

# ONNX -> TFLITE

subprocess.run(
    [
        "onnx2tf",
        "-i",
        ONNX_PATH,
        "-o",
        TF_DIR
    ],
    check=True
)

print("TFLite generated.")

# TFLITE -> HEADER

with open(TFLITE_PATH, "rb") as f:
    model_data = f.read()

with open(HEADER_PATH, "w") as f:
    f.write("#ifndef MODEL_H\n")
    f.write("#define MODEL_H\n\n")
    f.write("const unsigned char model[] = {\n")
    
    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            f.write("    ")
        f.write(f"0x{byte:02x},")
        if i % 12 == 11:
            f.write("\n")
            
    f.write("\n};\n\n")
    f.write(f"const unsigned int model_len = {len(model_data)};\n\n")
    f.write("#endif\n")

print(f"Header saved to: {HEADER_PATH}")

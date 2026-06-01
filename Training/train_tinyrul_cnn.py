import os
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAT_FILE = "Oxford_Battery_Degradation_Dataset_1.mat"

ALL_CELLS = [
"Cell!"
"Cell2",
"Cell3",
"Cell4",
"Cell5",
"Cell6",
"Cell7",
"Cell8"
]

SEQ_LEN = 10
EPOCHS = 80
BATCH_SIZE = 16
INPUT_FEATURES = 11

print("Using Device:", DEVICE)

def extract_cell_data(cell_struct):

```
cycle_names = sorted(
    [name for name in cell_struct[0,0].dtype.names
     if "cyc" in name]
)

all_features = []

for cyc in cycle_names:

    try:

        cycle_data = cell_struct[0,0][cyc][0,0]
        discharge = cycle_data["C1dc"][0,0]

        t = discharge["t"].flatten()
        v = discharge["v"].flatten()
        q = discharge["q"].flatten()
        T = discharge["T"].flatten()

        min_len = min(len(t), len(v), len(q), len(T))

        t = t[:min_len]
        v = v[:min_len]
        q = q[:min_len]
        T = T[:min_len]

        duration = t[-1] - t[0]

        capacity = np.abs(q[-1] - q[0])

        energy = np.trapezoid(
            v * np.abs(q),
            t
        )

        avg_temp = np.mean(T)
        max_temp = np.max(T)

        voltage_mean = np.mean(v)
        voltage_std = np.std(v)

        dv = np.diff(v)
        dq = np.diff(q)

        valid = np.abs(dv) > 1e-5

        dv = dv[valid]
        dq = dq[valid]

        if len(dv) > 10:

            dqdv = savgol_filter(
                dq / dv,
                11,
                3
            )

            dqdv_peak = np.max(np.abs(dqdv))
            dqdv_area = np.trapezoid(np.abs(dqdv))

        else:

            dqdv_peak = 0
            dqdv_area = 0

        hist, _ = np.histogram(
            v,
            bins=30,
            density=True
        )

        entropy = -np.sum(
            (hist + 1e-8) *
            np.log(hist + 1e-8)
        )

        features = [
            capacity,
            energy,
            duration,
            avg_temp,
            max_temp,
            voltage_mean,
            voltage_std,
            dqdv_peak,
            dqdv_area,
            entropy
        ]

        all_features.append(features)

    except:
        continue

return np.array(all_features)
```
class BatteryDataset(Dataset):

```
def __init__(self, X, y):

    self.X = torch.tensor(
        X,
        dtype=torch.float32
    )

    self.y = torch.tensor(
        y,
        dtype=torch.float32
    )

def __len__(self):
    return len(self.y)

def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```
class TinyRULCNN(nn.Module):

```
def __init__(self, input_features):

    super().__init__()

    self.features = nn.Sequential(
        nn.Conv1d(
            input_features,
            16,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm1d(16),
        nn.ReLU(),

        nn.Conv1d(
            16,
            32,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm1d(32),
        nn.ReLU()
    )

    self.regressor = nn.Sequential(
        nn.Linear(32,32),
        nn.ReLU(),

        nn.Dropout(0.2),

        nn.Linear(32,16),
        nn.ReLU(),

        nn.Linear(16,1)
    )

def forward(self, x):

    x = x.permute(0,2,1)

    x = self.features(x)

    x = torch.mean(x, dim=2)

    x = self.regressor(x)

    return x.flatten()
```
print("Loading dataset...")

mat = sio.loadmat(MAT_FILE)

print("Extracting features...")

all_features = {}

for cell in ALL_CELLS:

```
all_features[cell] = extract_cell_data(
    mat[cell]
)
```

scaler = StandardScaler()

scaler.fit(
np.vstack(
[all_features[c] for c in ALL_CELLS]
)
)

X_train = []
y_train = []

for cell in ALL_CELLS:

```
feats = scaler.transform(
    all_features[cell]
)

cycle_idx = (
    np.arange(len(feats))
    / len(feats)
).reshape(-1,1)

feats = np.hstack(
    [feats, cycle_idx]
)

rul = np.arange(
    len(feats)-1,
    -1,
    -1
)

for i in range(
    len(feats) - SEQ_LEN
):

    X_train.append(
        feats[i:i+SEQ_LEN]
    )

    y_train.append(
        rul[i+SEQ_LEN]
    )
```

X_train = np.array(X_train)
y_train = np.array(y_train)

y_max = np.max(y_train)

y_train_norm = y_train / y_max

print("Training Samples:", len(X_train))
print("Input Shape:", X_train.shape)


loader = DataLoader(
BatteryDataset(
X_train,
y_train_norm
),
batch_size=BATCH_SIZE,
shuffle=True
)

model = TinyRULCNN(
INPUT_FEATURES
).to(DEVICE)

criterion = nn.HuberLoss()

optimizer = torch.optim.Adam(
model.parameters(),
lr=0.001
)

print("\n Training. \n")

model.train()

for epoch in range(EPOCHS):

```
running_loss = 0

for X_batch, y_batch in loader:

    X_batch = X_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    optimizer.zero_grad()

    pred = model(X_batch)

    loss = criterion(
        pred,
        y_batch
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    running_loss += loss.item()

if (epoch + 1) % 10 == 0:

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss = {running_loss/len(loader):.6f}"
    )
```

torch.save(
{
"model_state_dict":
model.state_dict(),

```
    "scaler_mean":
        scaler.mean_,

    "scaler_scale":
        scaler.scale_,

    "ymax":
        y_max,

    "input_features":
        INPUT_FEATURES
},
"RUL_TINYML.pth"
```

)

print("\nTraining Complete")


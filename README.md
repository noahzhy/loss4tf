
# Loss function in tf

Various loss function in tf, some of them were implemented by official tensorflow, but I prefer to pack them into a Layer class.

## Supported Loss Functions

### Recognition

- [x] CTC Loss
- [x] Focal CTC Loss
- [x] CTC Center Loss
   - [x] pytorch version from [here](./torch_losses.py)

#### CTC Center Loss comparison

```python
from unit_test import test_ctc_center_loss

n_class = 100
dims = 128
x = np.random.normal(size=(32, 16, dims)).astype(np.float32)
labels = np.random.randint(0, n_class, (32, 16)).astype(np.int32)
test_ctc_center_loss(x, labels, n_class, dims)
```

results:

```bash
torch loss: 127.31403
tf    loss: 127.314026
loss  diff: 7.6293945e-06
```

### Classification

- [x] BCE Loss
- [x] CE Loss
- [x] Center Loss

### Object Detection

- [x] Smooth L1 Loss

### Segmentation

- [x] Dice BCE Loss
- [x] Dice Loss
- [x] IoU Loss

## Usage

```python
from losses import *

# e.g. CTC Loss
loss = CTCLoss()
x = tf.random.normal([2, 10, 20])
y = tf.random.uniform([2, 10], maxval=20, dtype=tf.int32)
out = loss(x, y)
print(out)
```

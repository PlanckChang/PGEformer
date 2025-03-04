# A Spatial Node is Worth a Temporal Patch: Patch Granularity Efficient Transformer for Long-term Traffic Prediction

We have organized the codebase and released the core part for review.
## Installation

- python = 3.10.10
- pip install -r requirements.txt

'''
numpy==1.24.3
pandas==2.2.1
PyYAML==6.0.1
torch==2.0.0
torchinfo==1.8.0
'''

## Usage

```python
nohup python train.py -i 48 -o 48 -d pems04 -g 0 -m for_train &
```

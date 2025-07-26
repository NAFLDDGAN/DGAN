# DGAN


## Overview
This repository contains the implementation of the DGAN model using PyTorch. The model is specifically designed for diagnosing NAFLD in T2DM patients and is applied in the paper titled "**[Double Graph Attention Network for Predicting Non-alcoholic Fatty Liver Disease in Patients With Type 2 Diabetes]**" For detailed information, please refer to the paper.

## Paper Reference
If you use or refer to this DGAN model in your work, please cite the following paper:
"**[Double Graph Attention Network for Predicting Non-alcoholic Fatty Liver Disease in Patients With Type 2 Diabetes]**"

## Requirements
PyTorch,NumPy,PyTorch Geometric

## Example Usage
```python
# Importing the DGAN model
from DGAN import DGAN

# Creating an instance of the DGAN model
model = DGAN()

...

# Forward pass
output = model(feature)

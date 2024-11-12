## Overview
Here we propose a robust spatio-temporal fusion deep learning model to capture the spatio-temporal relationship between evapotranspiration (ET) and input surface properties. In addition, a coarse-to-fine two-stage training strategy is introduced to enable the model learn general patterns from widely used ET products in the first stage and refine model parameters with flux tower observations in the second stage.
## Model Architecture
The framework of the model is illustrated in the figure below, and the specific implementation can be found in **model.py**. This neural network model is designed to predict ET spatiotemporal data sequences, which adopts an encoding-decoding network architecture that comprises feature fusion, spatial encoder, temporal encoder, and decoder modules
![模型结构图_modify](https://github.com/user-attachments/assets/b3e6a49a-f9bd-41d1-be36-f8d91fa209d7)
## File Structure
- `model.py`: Model inplementation, defining the **ETModel** class.
- `dataset.py`: Defines the **ETDataset** class for loading and processing input features and targets from .npy files. It handles data slicing by temporal and spatial parameters, supports data augmentation through optional perturbations, and applies masking for missing values.
- `train.py`: Handles the training and validation process for **ETModel**, using **ETDataset** to load data and applying masking for invalid points. It defines a custom loss function and saves model checkpoints periodically, recording metrics across training epochs.
- `test.py`: Evaluates **ETModel** on test data by calculating root mean square deviation (RMSD) and correlation coefficient, using masking to handle invalid data points. Results are saved for comparative analysis between model predictions and ground truth.
- `.gitignore`: Excludes large data files (e.g., .npy) to keep the repository within GitHub's size limits.
- `README.md`: Project documentation

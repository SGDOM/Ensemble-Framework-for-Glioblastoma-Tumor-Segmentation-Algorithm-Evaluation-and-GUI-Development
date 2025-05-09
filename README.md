# Ensemble-Framework-for-Glioblastoma-Tumor-Segmentation-Algorithm-Evaluation-and-GUI-Development


## Required environment
Install the requirements.txt dependencies

```
pip install -r requirements.txt
```

## Model Training and prediction
Dataset preparation

#### Datasets

BraTS 2021:http://braintumorsegmentation.org/. 

BraTS 2020: https://www.med.upenn.edu/cbica/brats2020/data.html

BraTS 2018: https://www.med.upenn.edu/sbia/brats2018/data.html

MSD dataset: http://medicaldecathlon.com/


#### Dataset preprocessing and Load required functions

Running the Preprocessing.py

```
python Preprocessing.py
```

#### Model training and prediction

Running the Training & prediction.py
```
python Training & prediction.py
```
hyperparameters like learning rate and Focal Tverskey loss parameters varies over 50 epochs for swift convergence.

#### GUI development

Running the app.py
```
python app.py
```

#### Simulation Results
Simulation findings including generalization are disclosed subsequent to the paper's publication.

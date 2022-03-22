nasa_air_quality_competition
==============================

The project deals with the competition in Professor Iddo Drori class. In the project, we will be predicitng the pm2.5 level for the competition. 

# Setup:
```
aws s3 cp s3://drivendata-competition-airathon-public-us/pm25/train/maiac train/maiac --no-sign-request --recursive
aws s3 cp s3://drivendata-competition-airathon-public-us/pm25/test/maiac test/maiac --no-sign-request --recursive

wget "https://drivendata-prod.s3.amazonaws.com/data/88/public/train_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYZDRLDSRZ%2F20220319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220319T204811Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4ebc052c72e2aad8a9b6da7db05e2dc24a7093f956c69d7f3dc41077174b451a" -O train_labels.csv

wget "https://drivendata-prod.s3.amazonaws.com/data/88/public/grid_metadata.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYZDRLDSRZ%2F20220319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220319T204811Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=247344fd24014332cd24d4e77b210ce66a1832ec6bbc851b7add08e6a0673e24" -O  grid_metadata.csv

wget "https://drivendata-prod.s3.amazonaws.com/data/88/public/pm25_satellite_metadata.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYZDRLDSRZ%2F20220319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220319T204811Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7029344ffb91a8c9e18e863512cc95288dd0cb13b92df63b9031cbe1d238cffe" -O pm25_satellite_metadata.csv


wget "https://drivendata-prod.s3.amazonaws.com/data/88/public/submission_format.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYZDRLDSRZ%2F20220319%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220319T204811Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4c0d884967288edbe48b144a27da0024974337ec553846e6fe29c60bebe64a1d" -O submission_format.csv
```

# Explanation
Our process involves basically moving from a higher reolsution 1200x1200 images to go into smaller frames. We basically use a VAE to move from a high demensional 128 vectors. The basic principle of the our approach is shrinking the size and then use other process to effectively use the results. 

In the final experiement, we use a FNN and a lightGBM model for the results. 

The FNN model is:

```
class Regression(pl.LightningModule):
    
    def __init__(self,NUM_FEATURES):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.f = nn.Linear(64, 1)
        self.activation = nn.LeakyReLU()
    

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.f(x)
        return x
```

The light GBM is 
```
params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 50,
        "learning_rate": 0.05,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

```

We also use a hierchial model to go through the results. 
# Experiments
We also had done a lot of experiments. Feel free to check the notebooks tabs for final uses. These include.
* Deep Vision Transofrmer
* Vision Transofrmer
* NBETS Model
* Regression Model
* FB Propehet
* Temporal Fusion model


# Navigation
All these files are in thh notebook folder with final experiments. The final models are in the src models

To setup and understand our preprocess steps please take a look at the notebooks at get_setup folder.

Since our dataset is to large, we have uploaded most of our data gdrive. If you need the access to processed data feel free to download it at. 

```
https://drive.google.com/file/d/19TSus9btTsgM4GaWaax-4_QITm8uUcOV/view?usp=sharing
```

Hope you like our work!



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>




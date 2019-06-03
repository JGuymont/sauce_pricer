# derivative_generator

Experiments on using generative models to learn the joint distribution of features used to price derivative and their market price

## Installation

In order to use the code, you will need to install the following dependencies:

- Pytorch
- Numpy
- scikit-learn
- matplotlib

```shell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## Project structure

Here is the explanation of folder structure:

- **_option_pricer_**: Stores the source code: data loader, preprocessor, models, experiments, and the script to train 
the different models.
    - **_models_**: Contans all the ML models: MLP regressor, Autoencoder, and Missing values imputer
    - **_data_**: Contains everything related to the preprocessing
    - **_utils_**: Contains utility function like the functions to make the plots
- **_tests_**: Unit tests
- **_trained_models_**: Folder for storing trained parameters.
- **_data_**: Contains all the datasets.
- **_config_**: Contains the hyperparameters of the different models.
- **_results_**: Contains the results of the experiment: figures, losses, predictions, etc.

## Running the experiments

The 4 main functions that are called when running the main script are:

- `train.cli(...)` in `option_pricer/train.py`
- `predict.cli(...)` in `option_pricer/predict.py`
- `visualize.cli(...)` located in `option_pricer/visualize.py`
- `split_data.cli(...)` located in `options_pricer/split_data.py`

To run one of these function:

```shell
usage: python -m option_pricer <command> [<args>]
Available commands:
   split_data            Split the data
   train                 Trains a model
   predict               Predicts using a pretrained model
   visualize             Visualizes experimental results
```

### Split the data

Start by splitting the data:

```shell
python -m option_pricer split_data
```

Running this script will split the data `PSPOption_RAW_Data.pkl` into 3 data sets:

- `./data/train_data.pkl` contains *complete data* from January 2017 to August 2017; 
- `./data/valid_data.pkl` contains *complete data* from September and October 2017; 
- `./data/test_data.pkl` contains *complete data* from November and December 2017.

*Complete data* means that only lines without missing values are kept.
Data from Jan to Au with missing values are saved under `./data/train_data_missing.pkl`.

### Training a model

```shell
usage: python -m option_pricer train [<args>]
 --train_path TRAIN_PATH
                        Path to the pickled training data set.
  --valid_path VALID_PATH
                        Path to the pickled validation data set.
  --data_path DATA_PATH
                        Path to the full pickled data set. This data will be
                        use only to list all the possible classes of the
                        categorical features.
  --model_save_dir MODEL_SAVE_DIR
                        Path where to save the trained parameters after
                        training.
  --results_dir RESULTS_DIR
                        Path of the directory where to save results (e.g. the
                        train/valid losses).
  --numerical_preprocessor_save_path NUMERICAL_PREPROCESSOR_SAVE_PATH
                        Path where to save the parameters of the numerical
                        preprocessor. The same parameters are going to be used
                        when making inference.
  --categorical_preprocessor_save_path CATEGORICAL_PREPROCESSOR_SAVE_PATH
                        Path where to save the parameters of categorical
                        preprocessor. The same parameters are going to be used
                        when making inference.
  --model_definition MODEL_DEFINITION
                        Path to model definition. Model definition is a json
                        file containing all the hyperparameters of the model.
```
Note that most of the arguments have default parameters so only the model definition is required.

##### Example: training the mlp regressor

```shell
python -m option_pricer train --model_definition config/regressor.json
```

Running this line of code in the terminal will train the mlp regressor with the hyperparameters 
in `config/regressor.json`. The default training and validation set are the one that is set in `option_pricer/globals.py`
under the variable `TRAIN_PATH` and `VALID_PATH`. After the model is trained, the weights of the model 
and the preprocessing parameters are saved in the directory `trained_models`.

##### Example: training the MLP with categorical features

```shell
python -m option_pricer train --model_definition config/regressor_mix_features.json
```

##### Example: training the autoencoder

```shell
python -m option_pricer train --model_definition config/autoencoder.json
```

to merge the missing value prediction to the train data, run

```shell
python -m option_pricer merge data/train_data.pkl results/predictions_missing_value_imputer_train_data_missing.pickle --out_path data/train_data_augmented.pkl
```

## Making predictions using trained models

```shell
usage: python -m option_pricer predict [<args>]
  --model_definition MODEL_DEFINITION
                        Path to json model definition
  --model_state_path MODEL_STATE_PATH
                        Path where to the trained parameters
  --data_path DATA_PATH
                        path to the pickled dataframe on which prediction
                        should be made
  --numerical_preprocessor NUMERICAL_PREPROCESSOR
                        Path of the saved numerical preprocessor
  --categorical_preprocessor CATEGORICAL_PREPROCESSOR
                        Path to the saved categorical preprocessor
  --output_directory OUTPUT_DIRECTORY
                        Path where to save the prediction of the experiment
```

##### Example: predicting prices on test data

```shell
python -m option_pricer predict --model_definition config/regressor.json --model_state_path trained_models/model_state_mlp_regressor.pickle --data_path data/train_data.pkl                 
python -m option_pricer predict --model_definition config/regressor.json --model_state_path trained_models/model_state_mlp_regressor.pickle --data_path data/valid_data.pkl                 
```

##### Example: Imputing the missing values and saving the results. The preprocessing in inverted after the prediction

```shell
python -m option_pricer predict --model_definition config/missing_imputer.json --model_state_path trained_models/model_state_autoencoder.pickle --data_path data/train_data_missing.pkl
```

## Visualizing the results

### Plot MSE Loss

```shell
python -m option_pricer visualize --mse_loss --writer_path results/writer_mlp_regressor.pickle --output_filename mlp_regressor_loss.png --output_dir results
```

## Running the test

```shell
python -m unittest tests.test_preprocessing
python -m unittest tests.test_missing_imputation
```

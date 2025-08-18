import os

# Default locations for training and test datasets.
# These can be overridden by setting the TRAIN_DATA_PATH and TEST_DATA_PATH
# environment variables before running the notebook.
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/application_train.csv')
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'data/application_test.csv')

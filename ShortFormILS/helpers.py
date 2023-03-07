import os
import random

import numpy as np


def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def round_to_class(y_bf):
    y_bf = np.where(y_bf == -5, 0, y_bf)
    y_bf = np.where(y_bf == 1, 2, y_bf)
    y_bf = np.where(y_bf == -1, 2, y_bf)
    y_bf = np.where(y_bf == -3, 1, y_bf)
    y_bf = np.where(y_bf == 3, 3, y_bf)
    y_bf = np.where(y_bf == 5, 4, y_bf)
    return y_bf

def round_to_value(y_bf):
    y_bf = np.where(y_bf == -3, -1, y_bf)
    y_bf = np.where(y_bf == -7, -3, y_bf)
    y_bf = np.where(y_bf == -5, -3, y_bf)
    y_bf = np.where(y_bf == -11, -5, y_bf)
    y_bf = np.where(y_bf == -9, -5, y_bf)
    y_bf = np.where(y_bf == 3, 1, y_bf)
    y_bf = np.where(y_bf == 7, 3, y_bf)
    y_bf = np.where(y_bf == 5, 3, y_bf)
    y_bf = np.where(y_bf == 11, 5, y_bf)
    y_bf = np.where(y_bf == 9, 5, y_bf)
    return y_bf
# rount_to_value in excel formula:
# =WENN(L2=-3; -1; WENN(A1=-7; -3; WENN(A1=-5; -3; WENN(A1=-11; -5; WENN(A1=-9; -5; WENN(A1=3; 1; WENN(A1=7; 3; WENN(A1=5; 3; WENN(A1=11; 5; WENN(A1=9; 5; A1))))))))))

def round_to_dim(y_bf):
    y_bf = np.where(y_bf <= 1, 0, y_bf)
    y_bf = np.where(y_bf == 2, 1, y_bf)
    y_bf = np.where(y_bf >= 3, 2, y_bf)
    return y_bf

def round_to_critical_cases_classes(y_bf):
    # they are already processed by round_to_class.
    # make the first and laste step half of the second and second last step
    y_bf = np.where(y_bf == 1, 0.5, y_bf)
    y_bf = np.where(y_bf == 2, 1.5, y_bf)
    y_bf = np.where(y_bf == 3, 2.5, y_bf)
    y_bf = np.where(y_bf == 4, 3, y_bf)
    return y_bf
# python_helpers

Some helper modules in python designed to help clean up code, or perform simple functions that couldn't be found in existing python packages.

## Table of Contents
### binning_helpers

Some functions designed to help bin data. Includes functions to:
* create bins of data equal in number of data points rather than width
* bin/plot data by 1,2, or 3 parameters

### Regressors

Some functions designed to aid in using scikit-learn machine learning models.Mostly designed to reduce lines of code needed in a main script. Includes functions to:
* create X,y feature matricies
* get the accuracy of a model (with various options for determining what "accurate" means)
* get the predictions from a model, with some measure of the "accuracy" of each individual prediction, also given various options for defining accurate.

## Installing

Modules should be ready to go as-is. They can be downloaded and imported intopython using (for example):

```
import binning_helpers as binners
import Regressors as reg
```

# Text Classification

Multi-model implementation for text classification.

## Commands

Install required fundamental packages:

```
pip3 install pandas
pip3 install torch
pip3 install gensim
```

If you wish to go with the same dataset as here, follow the guidelines [here](https://github.com/arunism/Text-Classification/blob/master/dataset/README.md).

Prepare your dataset with the following command:

`python3 dataset/dataset.py`

Train your model with:

`python3 train.py`


## Configuration

You can always manage your project configuration using the file `constants.py`.

`Remember:` This should be done before training your model or it may not work as expected.
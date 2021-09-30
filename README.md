# Sound-Anomaly-Detection
Extract mel spectrogram from wave, train them, test the trained model

## Installation

### Preprocessing
extract spectrogram
> **inspection/parameters.py**: Modify parameters using in preprocessing.

> **inspection/model_info.py**: Modify classes. These will be used when calculating metrics and training

> **inspection/preprocess_fns.py**: Create your own preprocess method and add its parameters to parameters.json

### Training
* `tdms` files need to be prepared
* file name sould be 'date_label_file name.extension'
  * ex) 210930-GOOD-sample1.tdms
  * '-' is a delimiter. you can change it at `consts.py`

> **models/**: Trained model will be saved in this directory.

```shell
$ python train.py
```

### Test
`tdms` files need to be prepared
```shell
$ python test.py
```
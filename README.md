# maliciousfl - Offline
In so far, our HDBSCAN cannot support secure multi-party computation (smpc). If you want to use smpc, please go to [offlineCrypto](https://github.com/Timo9Madrid7/maliciousfl/tree/offlineCrypto) instead. 

## Dependencies
- In the `requirements.txt`:
  - sympy
  - mpmath
  - scipy
  - matplotlib
  - numpy
  - options
  - Pillow
  - PyYAML
  - scikit_learn
  ```python
  pip install -r requirements.txt
  ```
- Other:
  ```python
  pip install autodp hdbscan torch_summary torchsummary
  pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```

## Dataset
- [x] MNIST
- [x] CIFAR-10
- [x] EMNIST

### How to split the data?
- MNIST:
  ```python
  python Common/Utils/data_splitter.py --dataset=MNIST --unique=True
  ```
  Other `args` can be found by run `python Common/Utils/data_splitter.py --help`
- CIFAR-10:
  ```python
  python Common/Utils/data_splitter.py --dataset=CIFAR10 --unique=True
  ```
  Other `args` can be found by run `python Common/Utils/data_splitter.py --help`
- EMNIST:
  ```python
  python Data/EMNIST/generate_data.py 
  ```
  Other `args` can be found by run `python Data/EMNIST/generate_data.py --help`

## Configuration
Create your own folder with a configuration file under folder `Experiments/`. 

An example is provided in `Experiments/config_default`.

All the historical results, e.g. main task accuracy, will be stored in the folder you create

## Available attacks
- [x] backdoor
- [x] flipping
- [x] edge-case
- [x] Krum
- [x] trimmed-mean
- [x] random upload

## Run simulations
```python
python model_prepare.py --config=<path_to_your_folder>
# python model_prepare.py --config=Experiments/config_default/

python fl_offline.py
```
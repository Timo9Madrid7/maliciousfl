# maliciousfl

## Start

### CrypTen (Linux Only)
~~If choosing `CrypTen`, make sure to comment `crypten/init.py` from `line 194` to `202` before running `python fl_offline_crypten.py`, since it currently doesn't support GPU random generator.~~
If choosing `CrypTen`, make sure to pass argument `device=cpu` when initializing `crypten.init()`, since it currently doesn't support GPU random generator on **Google CoLab**.

In addition (not necessary), 
- ~~comment `line 75` in `crypten/encoder.py` and add a new line of code below it: `dividend = tensor.div(self._scale, rounding_mode='trunc') - correction` so as to avoid `UserWarning: __floordiv__ is deprecated`~~
- ~~comment `line 27` and add a new line of code below it: `quotient = tensor.div(integer, rounding_mode='trunc')` so as to avoid `UserWarning: __floordiv__ is deprecated`~~
- add `warnings.filterwarnings("ignore", message="__floordiv__")` to `crypten/encoder.py`

Open cmd and run commands:
```cmd
python model_prepare.py && python fl_offline_crypten.py
```

v2 can run on Google CoLab: [link here](https://colab.research.google.com/drive/1DBlVAJr6CWOmcrbLS5i7-lnJJafsD0-f?usp=sharing)

### SyMPC + PySyft
If choosing `SyMPC`, `torch` version should lower than `1.8.1`.

Open cmd and run commands:
```cmd
python model_prepare.py & python fl_offline_sympc.py
```

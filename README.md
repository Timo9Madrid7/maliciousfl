# maliciousfl

## Start

### CrypTen
If choosing `CrypTen`, make sure to comment `crypten/init.py` from `line 194` to `202` before running `python fl_offline_crypten.py`, since it currently doesn't support GPU random generator.

In addition, comment `line 75` in `crypten/encoder.py` and add a new line of code below it: `dividend = tensor.div(self._scale, rounding_mode='floor') - correction` so as to avoid `UserWarning: __floordiv__ is deprecated` (not necessary)

```cmd
python model_prepare.py & python fl_offline_crypten.py
```

### SyMPC + PySyft
If choosing `SyMPC`, `torch` version should lower than `1.8.1`.

```cmd
python model_prepare.py & python fl_offline_sympc.py
```

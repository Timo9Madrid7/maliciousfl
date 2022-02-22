# maliciousfl

## Start

### CrypTen
If choosing `CrypTen`, make sure to comment `crypten/init.py` from `line 194` to `202` before running `python fl_offline_crypten.py`, since it currently doesn't support GPU random generator.

```cmd
python model_prepare.py & python fl_offline_crypten.py
```

### SyMPC + PySyft
If choosing `SyMPC`, `torch` version should lower than `1.8.1`.

```cmd
python model_prepare.py & python fl_offline_sympc.py
```

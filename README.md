# maliciousfl - Offline


## Data split
To modify `Common\Utils\data_splitter.py`:
  - dataset
  - total number of clients
  - number of samples per client
  - Non-IID degree

and run command:
```
python Common\Utils\data_splitter.py
```

## Start FL training

To modify `OfflinePack\offline_config.py`:
  - global/local setting
  - attacking setting
    - malicious random uploads
    - label flipping attack
    - trigger backdoor attack
    - edge case attack
    - Krum selection attack
    - trimmed mean attack

and run commands
```python
python model_prepare.py # print model info
python fl_offline.py # start training
```
#!/bin/sh
for i in {0..8}; do
    python ppfl_client.py --id $i &:
done &
python ppfl_client.py --id 9
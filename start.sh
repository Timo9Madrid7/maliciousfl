#!/bin/sh
for i in {0..8}; do
    python clear_dense_client.py --id $i &
done &
python clear_dense_client.py --id 9
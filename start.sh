#!/bin/sh
for i in {0..8}; do
    python clearflguard_client.py --id $i &
done &
python clearflguard_client.py --id 9
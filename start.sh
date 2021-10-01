#!/bin/sh
for i in {0..8}; do
    python clearkrum_client.py --id $i &
done &
python clearkrum_client.py --id 9
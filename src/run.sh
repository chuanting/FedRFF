#!/bin/bash
python centralized_baseline.py &
python centralized_metric.py ;
python centralized_adapt.py &
wait
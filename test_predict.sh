#!/bin/bash

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a quick test fake job ad. Earn $5000 a week!"}'

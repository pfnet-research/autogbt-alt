#!/bin/bash
set -euo pipefail

docker run -i --rm \
           -v $(pwd):/autogbt \
           -w /autogbt \
           python:$PYTHON_VERSION \
           bash << EOF
set -euo pipefail
pip install .
pip install -r requirements-dev.txt
./test.sh
python example/breast_cancer.py --n-trials 1
python example/boston.py --n-trials 1
python example/custom_objective.py
EOF

#!/usr/bin/env bash
python3 -m coverage run --source=. --omit=tests/* -m pytest . && echo "Tests Passed" || exit 1
python3 -m coverage lcov -o coverage/lcov.info --ignore-errors || true

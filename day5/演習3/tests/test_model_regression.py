name: Test

on:
  push:
    branches:
      - main
      - develop
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Extract baseline model from main
        id: baseline
        env:
          branch: main
        run: |
          if git cat-file -e "origin/${branch}:day5/演習3/models/titanic_model.pkl" 2>/dev/null; then
            git show "origin/${branch}:day5/演習3/models/titanic_model.pkl" > /tmp/baseline_model.pkl
          else
            echo "No baseline model found in main branch."
          fi

      - name: Run tests
        env:
          HAS_BASELINE: ${{ steps.baseline.outputs.has_baseline }}
          BASELINE_MODEL: /tmp/baseline_model.pkl
        run: |
          pytest tests/
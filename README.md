# Find My Major

A tool for matching student interests to areas of study.

## Getting Started

1. Set up python environment:

```
conda env create --file environment.yml
conda activate csc-570
```

## Project Layout

- `embeddings`: Sklearn-style transformers that encode natural language into latent embedding vectors.
- `classifers`: Model architectures for classifying college majors.
- `test.py`: Evaluation and demo code for all models.
- `train.py`: Training loops for models.
- `helper.py`: Utility methods for loading and preprocessing data

## Command Line Demo

Run the `demo()` function in `test.py`

## Web Server

The web server provides a RESTful API for getting major recommendations:

```
python api.py
curl --location --request POST 'http://127.0.0.1:5000/recommend' \
--header 'Content-Type: application/json' \
--data-raw '{
    "data": "Construction is cool!"
}'

...

[
    "Construction Management",
    "Architectural Engineering",
    "City and Regional Planning"
]
``
```

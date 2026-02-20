# Can RAG pass an AI class? Evaluating question answering on lecture transcripts

This project shows whether RAG can answer exam-style questions based on lecture transcripts. The goal is to evaluate the performance and limitations of retrieval compared to local LLMs. The evaluation is based on a manually curated MCQA dataset, and accuracy is used as the main metric.

## Project structure

```
.
├── configs/
├── figures/
├── notebooks/
├── prompts/
├── pyproject.toml
├── ragqa/
├── README.md
├── requirements.txt
└── scripts/
```

## Execution Environment

All experiments were conducted locally on a dedicated workstation with the following specifications:

- CPU: Intel Core i5-12600KF (12th Gen)
- GPU: NVIDIA RTX 3090
- RAM: 64 GB
- OS: Linux
- Python: 3.10.16

## Setup and installation

### 1. Clone the repository

```
git clone https://github.com/alessiopittiglio/bdtm-project.git
cd bdtm-project
```

### 2. Create environment and install dependencies

We recommend using a virtual environment (e.g., Conda or venv):

```
# Using Conda
conda create -n bdtm_env python=3.10
conda activate bdtm_env

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Data preprocessing


Raw transcripts are cleaned before being used in the pipeline:

```
python clean_transcripts.py
```

## Vector database

After preprocessing, build the vector store:

```
python build_vector_store.py \
  --collection lectures \
  --config configs/retrieval/default.yaml
```

## MCQA dataset generation

To generate the dataset used for evaluation:

```
python generate_mcqa.py
```

The main evaluation script is:


## Evaluation

Run all experiments:

```
python run_evaluation.py
```

Run a specific experiment:

```
python run_evaluation.py --exp experiment_name
```

## Needle in a Haystack (NIAH)

To create incremental datasets for Needle-in-a-Haystack testing:

```
python generate_niah_dataset.py --sizes 1 5 10 20 50 --position middle
```

Run the evaluation:

```
python run_niah_evaluation.py
```

## Models 

Local models are executed via the Python bindings of **llama.cpp**, using the maintained version by Andrei Betlen.

The Python binding must be installed according to your system configuration (OS, Python version, CUDA).

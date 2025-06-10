# Lightweight Retrieval-Augmented Generation for Contract Question-Answering on Local Hardware

This is the repo for publication. It is not the main repo for development. It contains only the code necessary to run the experiments ([`cbfk/`](./cbfk/)) and generate the results reported in the paper ([`insights/`](./insights/)). Because of the proprietary nature of the corpus, the data and the experiment artefacts are not included in this repo. Also not included are test modules, deployment scripts, etc. to mitigate the risk of accidental disclosure of sensitive information.

For the paper itself, along with short, medium, and long versions of a NotebookML-generated podcast about the paper, see [`paper/`](./paper/).

While the exact experiments from the paper cannot be reproduced due to the proprietary corpus, the following commands can be used to run the workflow on your own corpus:

```bash
# Create a new sweep
uv run cbfk/sweep_config.py --wandb-project my-project-name

# Download the sweep results from wandb
uv run insights/etl_db.py --wandb-project my-project-name

# Create plots and tables
uv run insights/db_plot.py
```
The corpus should be markdown files in folder ``corpus/Contract``, the ground truth is ``corpus/Contract Ground Truth.xlsx`` with one sheet and columns Question, Answer, File, Section, Quote.

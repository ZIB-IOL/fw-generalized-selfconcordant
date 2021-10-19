# Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions

Repository for the paper "Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions" accepted at NeurIPS 2021.

Find the preprint [here](https://arxiv.org/abs/2105.13913).

Due to their large size, the raw instance data files are not included in the repository but available on the [Zenodo](https://doi.org/10.5281/zenodo.4836008) archive.
Run the `get_data_instances.sh` bash script (or equivalent on your system) to fetch them.
If `wget` or `unzip` are not available for you, download `data.zip` from the URL and place the `data` folder at the top-level of this repository.
`plotting/plot_experiment_results.py` is used to generate the figures stored in `Images/`.
Repository structure:

```
├── data # raw instance data obtained from get_data_instances.sh
├── Images # Final images
├── plotting # Plotting script
└── results # result JSON files
```

All top-level scripts generate data or results.

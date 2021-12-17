# GCDF1

This repository contains the source code for the user model evaluator proposed in the
[GCDF1: A Goal- and Context- Driven F-Score for Evaluating User Models](https://aclanthology.org/2021.eancs-1.2.pdf)
along with guidance on how to reproduce the results presented in our paper.

Currently, the evaluator only supports MultiWOZ 2.1 dataset and there are no plans to extend to other MultiWOZ versions
or SGD unless there is community interest in doing so - see the [Invited Contributions](#invited-contributions) section
for details about this project.

## Installation

This assumes you have `anaconda3` or `miniconda3` installed on your system. To set up the environment

1. run the command below to create the virtual environment
   ```
   conda env create -f environment.lock.yml
   ```
2. activate the new environment with:
   ```
   conda activate gcdf1
   ```

> **_NOTE:_**  The conda environment will have gcdf1 installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


For those interested in contributing to the project, needed only once after `git clone`:


3. update the environment to contain packages needed during development
   ```bash
   conda env update -f dev_environment.yml --prune
   ```
4 .install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes getting_started/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

## Running the evaluator
We provide conversations generated according to the setup described in Section 4.1 of our paper
for the MultiWOZ 2.1 test set in the `models/convlab_baselines/test` directory.

The evaluator can be run with the `evaluate` command after the environment has been activated, so type
`evaluate --helpfull` and checkout the `gcdf1.evaluate_user` section to understand the CLI for our evaluator. The
command to generate the `baseline.json` file in `models/convlab_baselines` folder is

```bash
evaluate --prediction_dir models/convlab_baselines/test -o baseline.json -c configs/multiwoz_user_evaluator.yaml -cmap resources/multiwoz21_canonical_map.json
```

and should be run from the repository root.

## Conversations format

To familiarise yourself with the conversation format, have a close look at the conversations
inside the `resources/sample_conversation.json` folder. These conversations largely follow the
[Schema-guided Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
format here. We obtained these conversations in two steps:

- Obtain conversations between a `Convlab2` system and user model in the same format as MultiWOZ 2.1
- Convert the conversations to SGD format using a modified version of the `create_data_from_multiwoz.py` script provided [here](https://github.com/google-research/google-research/tree/master/schema_guided_dst/multiwoz)

Our script added the following information to  each dialogue:

   1. `final_goal`: dict storing the final goal after `Convlab2` processing. This is NOT used by the evaluator
   2. `goal`: dict, storing the dialogue goal, in the same format as the `Convlab2` goal model for MultiWOZ 2.1
   3. `metrics`: metrics computed by `Convlab2` for the conversation these are NOT used by the evaluator

We also added the following field to each turn:

   1. `nlu`. This field contains a single key `frames` with the same structure as the turn (i.e., an `actions` key which stores a list of actions recognised by the agent and a `service` key which indicates the domain). The evaluator will still work if this is not in the schema - NLU validation features should be disabled in the evaluator config in this case.

The evaluator outputs a complex hierarchical structure containing the behaviours detected for each metric. See the
documentation of the `gcdf1.utils.multiwoz_output.py` module for an overview of the behaviours.

## Reproducing our results
Make sure you have activated the `gcdf1` environment and install `jupyter` with the command

```bash
pip install jupyter
```

Start a jupyter notebook [in your browser](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) from the root. Then
navigate to the `getting_started` folder and run the `IGCDF1.ipynb` and `RGCDF1.ipynb` notebooks.

## Invited contributions

### Extending the library to other datasets

When developing the library, we recognised the need to keep the implementation general to ensure compatibility with all
MultiWOZ versions and other datasets. However, for pragmatic reasons, not all the implementation is agnostic to MultiWOZ
and minor refactoring is required so that some of the functions work correctly irrespective of the dataset. If you are
interested to help extend the framework review the code and get in touch - the code author is happy to guide and support
your endeavour and thus provide the community with a tool to reliably measure the performance of their user models.

### Adding NLG metrics

The canonical map for the MultiWOZ 2.1 version contains over 10000 value paraphrases extracted from the corpus. It
is thus straightforward to implement a more robust slot-error rate metric. The author is happy to provide basic code for
this and implementation guidance. The starting point could be the `value_generated` function implemented in the
`gcdf1.utils.evaluator` module. Integration of NLG metrics proposed in [this paper](https://aclanthology.org/2021.gem-1.4/)
is also desired.

Any SER-like metric can be integrated in the current framework to compute F1 scores based on natural language as opposed
to generated actions - get in touch to discuss this!

### Testing the library

Some basic test examples are included in the `tests/` folder. The library contains various `TODOs` with functions that
are not unit-tested. If the environment has been updated with the development dependencies as detailed in the
[installation](#installation) section, then you can run the tests from the root by simply invoking `tox` or `py.test`
commands.

# Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory containing the evaluator config files.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── dev_environment.yml     <- The conda environment file containing developer dependencies
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries. Each model has its own folder (e.g., convlab_baselines)
│                              and the test/dev/train subdirectories are expected in each model directory.
├── getting_started         <- Jupyter notebooks that can be run to reproduce paper results.
├── pyproject.toml          <- Build system configuration. Do not change!
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── resources               <- Contains the MultiWOZ2.1 canonical map file (multiwoz21_canonical_map.json) and the
│                              json file that it was generated from (multiwoz21_raw_canonical_map.json) using
│                              the script scripts/preprocess_multiwoz_canonical_map.py
├── scripts                 <- This contains the get_multiwoz21_raw_canonical_map.py script
│                              that can be used to generate a new "raw" canonical map and the
│                              preprocess_multiwoz_canonical_map.py which can be used to convert the raw canonical map
│                              to a format compatible with the evaluator.
├── setup.cfg               <- Declarative configuration of the project.
├── setup.py                <- Use `pip install -e .` to install for development or
│                              or create a distribution with `tox -e build`.
├── src
│   └── gcdf1               <- Actual Python package where the evaluator is implemented
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

# Citation

If you use any of the code in this repository or any of our methods, please cite our paper:

```
@inproceedings{coca-etal-2021-gcdf1,
    title = "{GCDF}1: A Goal- and Context- Driven {F}-Score for Evaluating User Models",
    author = "Coca, Alexandru  and
      Tseng, Bo-Hsiang  and
      Byrne, Bill",
    booktitle = "The First Workshop on Evaluations and Assessments of Neural Conversation Systems",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eancs-1.2",
    pages = "7--14",
    abstract = "The evaluation of dialogue systems in interaction with simulated users has been proposed to improve turn-level, corpus-based metrics which can only evaluate test cases encountered in a corpus and cannot measure system{'}s ability to sustain multi-turn interactions. Recently, little emphasis was put on automatically assessing the quality of the user model itself, so unless correlations with human studies are measured, the reliability of user model based evaluation is unknown. We propose GCDF1, a simple but effective measure of the quality of semantic-level conversations between a goal-driven user agent and a system agent. In contrast with previous approaches we measure the F-score at dialogue level and consider user and system behaviours to improve recall and precision estimation. We facilitate scores interpretation by providing a rich hierarchical structure with information about conversational patterns present in the test data and tools to efficiently query the conversations generated. We apply our framework to assess the performance and weaknesses of a Convlab2 user model.",
}
```

## Note

This project has been set up using [PyScaffold] 4.0.1 and the [dsproject extension] 0.6.1.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject

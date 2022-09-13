# factuality_bert

This repository contains code and data release for the following paper:

[Nanjiang Jiang, Marie-Catherine de Marneffe. He thinks he knows better than the doctors: BERT for event factuality fails on pragmatics. TACL 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00414/107616/He-Thinks-He-Knows-Better-than-the-Doctors-BERT)

### Getting started

First, follow the [tutorial](tutorials/setup_tutorial.md) to install dependencies and set up environments.

The directory [factuality_scripts](factuality_scripts) contains all the scripts to train the models experiments in the paper. They can be run as follows:
```
./factuality_scripts/factuality_single_task.sh CB 42
```

### Data
The directory [glue_data](glue_data) contains all the preprocessed data and splits used in the paper.

 
### License

This package is based on [jiant v1](https://github.com/nyu-mll/jiant-v1-legacy).
`jiant` is released under the [MIT License](LICENSE.md). The material in the `allennlp_mods` directory is based on [AllenNLP](https://github.com/allenai/allennlp), which was originally released under the Apache 2.0 license.


### Citation
```
@article{jiang-de-marneffe-2021-thinks,
    title = "He Thinks He Knows Better than the Doctors: {BERT} for Event Factuality Fails on Pragmatics",
    author = "Jiang, Nanjiang  and
      de Marneffe, Marie-Catherine",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.64",
    doi = "10.1162/tacl_a_00414",
    pages = "1081--1097",
    abstract = "Abstract We investigate how well BERT performs on predicting factuality in several existing English datasets, encompassing various linguistic constructions. Although BERT obtains a strong performance on most datasets, it does so by exploiting common surface patterns that correlate with certain factuality labels, and it fails on instances where pragmatic reasoning is necessary. Contrary to what the high performance suggests, we are still far from having a robust system for factuality prediction.",
}
```

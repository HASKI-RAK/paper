# Development of a Shortform ILS

This repository contains the scripts that were used to conduct an explorative data analysis (EDA) and thereafter determined the set of items used in the final shortform. We employ exhaustive feature selection (EFS) by calculating an accuracy of the subset and compare it with the result of the original form. It is possible to employ ML models to figure different combinations, but more context must be considered. Only choosing a subset is best achieved by calculating the accuracy, since we know the relation of X and y.

Fore more details please reffer to our paper [Development of a Shortform ILS](https://www.researchgate.net/).

## Installation

The scripts are written in Python 3.10.10. We recommend to use a virtual environment to install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run the scripts, you can use the following commands:

```bash
python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

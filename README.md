# Recommendation Training


## Description

This is repository containing the Deep Learning models zoo trainer for various category recommendation problems such that:
* general recommendation
* user-based recommendation 
* content-based recommendation
* session-based recommendation (sequential recommender)

## Installation
#### For using `pytorch` framework. We create virtual environment which contains 'pytorch' framework
```commandline
cd ./<folder_of_recommend_repository>
conda env create -n <name of environment> -f <the conda environment path>
conda activate <name fo environment>
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install pytorch pytorch-cuda=11.6 -c conda -c nvidia
# Add a new kernel that have contain execute file from this enviroment
python -m ipykernel install --user --name=<name of environment>
# Trigger Jupyter Lab Server
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &!
```
Example:
```commandline
cd ./<folder_of_recommend_repository>
conda env create -n pytorch_env -f ./venv/pytorch_env.yml
conda activate pytorch_env
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install pytorch pytorch-cuda=11.6 -c conda -c nvidia
# Add a new kernel that have contain execute file from this enviroment
python -m ipykernel install --user --name=pytorch_env
# Trigger Jupyter Lab Server
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &!
```

## Usage
To training and watching experiment log for the recommendation methods
Run: 
```commandline
python run_training.py --model=NARM --dataset=yoochoose --config_files=<path to config file>
```

## Support
KVC AI team - KiotViet Technology Corporation 

## Roadmap
None

## Contributing
Just following the Installation instruction and reading the project struct  

## Authors and acknowledgment
KVC AI team - KiotViet Technology Corporation

## Reference
Inspired by **RecBole** 

## License
None

## Project status
In Development Progress
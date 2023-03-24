#! /bin/bash

function ifEmpty()
{
    local VAL=$1
    local OPT=$2

    [[ -z "${VAL}" || "${VAL}" =~ ^[[:space:]]*$ || "${VAL}" == -* ]] \
    && { echo -e "\n  ERROR: Missing argument for option: -${OPT}\n" >&2; exit 1; }
}

usage()
{
    echo "usage: <command> options:<n|f|p>"
}

while getopts 'n:f:p:h' OPTION;
do
  case "$OPTION" in
    n)
      ifEmpty "${OPTARG}" "${opt}"
      env_name=${OPTARG//[[:space:]]}
      ;;
    f)
      ifEmpty "${OPTARG}" "${opt}"
      env_file_path=${OPTARG//[[:space:]]}
      ;;
    p)
      ifEmpty "${OPTARG}" "${opt}"
      package_file_path=${OPTARG//[[:space:]]}
      ;;
    h)
      usage
      exit
      ;;
    esac
done

# Create Virtual Environment
conda env create -n $env_name -f $env_file_path
conda activate $env_name
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install pytorch pytorch-cuda=11.6 -c conda -c nvidia
pip instal $package_file_path
# Add kernel to Jupyter Lab
python -m ipykernel install --user --name=$env_name
# Trigger Jupyter Lab server
# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &!

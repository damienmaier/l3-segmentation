# l3-segmentation

## Packages installation
```
conda create -n segmenter -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 python=3.10 scikit-learn=1.1.1 matplotlib=3.5.2 pandas=1.4.3 seaborn=0.11.2
conda activate segmenter
pip install tensorflow==2.9.1 keras-tuner==1.1.3
```

## Config file
After cloning this repo, you will need to create a local copy of `config_template.py` named `config.py` at the root of the project tree.
You can edit the values stored in this file to fit the needs of your setup.
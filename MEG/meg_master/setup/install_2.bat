@echo off
SET CUDA_VERSION=%1
IF [%1]==[] SET CUDA_VERSION=cpu
SET CUDA_VERSION=cpu
SET TORCH_VERSION=1.6.0
SET TORCH_GEOMETRIC_VERSION=1.6.0

call conda remove --name meg2 --all -y
call conda create --name meg2 python=3.10 -y
call conda activate meg2

call conda install pip -y
@REM call conda install rdkit -c rdkit -y
call conda install tensorboard -y
call conda install -c conda-forge typer -y

ECHO Installing RDKit...
python -m pip install rdkit-pypi==2022.9.5

python -m pip install geomloss

ECHO Installing PyTorch Geometric dependencies...
python -m pip install --no-cache-dir torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
python -m pip install --no-cache-dir torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
python -m pip install --no-cache-dir torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
python -m pip install --no-cache-dir torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
python -m pip install torch-geometric==2.2.0


@REM call pip install -r ./../../../../requirements.txt
ECHO Installing data science packages...
python -m pip install numpy==1.23.5
python -m pip install pandas==1.5.2
python -m pip install scikit-learn==1.0.2
python -m pip install matplotlib==3.6.2

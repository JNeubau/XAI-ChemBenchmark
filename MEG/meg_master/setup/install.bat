@echo off
SET CUDA_VERSION=%1
IF [%1]==[] SET CUDA_VERSION=cpu
SET CUDA_VERSION=cpu
SET TORCH_VERSION=1.6.0
SET TORCH_GEOMETRIC_VERSION=1.6.0

call conda deactivate
call conda remove --name meg --all -y
call conda create --name meg python=3.7 -y
call conda activate meg

call conda install pip -y
call conda run -n meg pip install numpy=1.19.2 -y
@REM call conda install -n meg pip numpy=1.19.2 -y
call conda install rdkit -c rdkit -y
call conda install tensorboard -y
call conda install -c conda-forge typer pillow==8.2.0 matplotlib networkx -y
call conda install -n meg pytorch=%TORCH_VERSION%+cpu torchvision=0.7.0 cpuonly -c pytorch -y

call conda run -n meg pip install geomloss
call conda run -n meg pip install protobuf==3.19

@REM python -m pip install torch==%TORCH_VERSION%+cpu torchvision==0.7.0+cpu torchaudio==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
call conda run -n meg pip install torch==%TORCH_VERSION%+cpu -f https://download.pytorch.org/whl/torch_stable.html
@REM call conda install -n meg pytorch=%TORCH_VERSION%+cpu torchvision=0.7.0 cpuonly -c pytorch -y

@REM python -m pip install --no-cache-dir torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
@REM python -m pip install --no-cache-dir torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html

@REM python -m pip install --no-cache-dir torch-cluster==1.5.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
@REM python -m pip install --no-cache-dir torch-spline-conv==1.2.0 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
@REM python -m pip install torch-geometric==1.6.0
call conda run -n meg pip install --no-cache-dir torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
call conda run -n meg pip install --no-cache-dir torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
call conda run -n meg pip install --no-cache-dir torch-cluster==1.5.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
call conda run -n meg pip install --no-cache-dir torch-spline-conv==1.2.0 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
call conda run -n meg pip install torch-geometric==1.6.0
@REM call conda run -n meg pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM call conda run -n meg pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM call conda run -n meg pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM call conda run -n meg pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM call conda run -n meg pip install torch-geometric==%TORCH_GEOMETRIC_VERSION%
@REM call pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
@REM call pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
@REM call pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
@REM call pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.6.0+cpu.html
@REM call pip install torch-geometric==1.6.0
@REM call conda run -n meg pip install https://data.pyg.org/whl/torch-1.6.0%2Bcpu/torch_scatter-2.0.5-cp37-cp37m-win_amd64.whl
@REM call conda run -n meg pip install https://data.pyg.org/whl/torch-1.6.0%2Bcpu/torch_sparse-0.6.8-cp37-cp37m-win_amd64.whl
@REM call conda run -n meg pip install https://data.pyg.org/whl/torch-1.6.0%2Bcpu/torch_cluster-1.5.8-cp37-cp37m-win_amd64.whl
@REM call conda run -n meg pip install https://data.pyg.org/whl/torch-1.6.0%2Bcpu/torch_spline_conv-1.2.0-cp37-cp37m-win_amd64.whl
@REM call conda run -n meg pip install torch-geometric==1.6.0


@REM call pip install typer torch-geometric==1.6.0 torch-sparse==0.6.8 torch-scatter==2.0.5 torch-cluster==1.5.8 torch-spline-conv==1.2.0
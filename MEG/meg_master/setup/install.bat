@echo off
SET CUDA_VERSION=%1
IF [%1]==[] SET CUDA_VERSION=cpu
SET CUDA_VERSION=cpu
SET TORCH_VERSION=1.6.0
SET TORCH_GEOMETRIC_VERSION=1.6.0

call conda remove --name meg --all -y
call conda create --name meg python=3.7 -y
call conda activate meg

call conda install pip -y
call conda install rdkit -c rdkit -y
call conda install tensorboard -y
call conda install -c conda-forge typer -y

@REM IF "%CUDA_VERSION%"=="cpu" (
@REM call conda install pytorch==%TORCH_VERSION% torchvision torchaudio cpuonly -c pytorch -y
python -m pip install torch==%TORCH_VERSION%+cpu torchvision==0.7.0+cpu torchaudio==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
@REM ) ELSE IF "%CUDA_VERSION%"=="cu92" (
@REM     call conda install pytorch==%TORCH_VERSION% torchvision torchaudio cudatoolkit=9.2 -c pytorch -y
@REM ) ELSE IF "%CUDA_VERSION%"=="cu101" (
@REM     call conda install pytorch==%TORCH_VERSION% torchvision torchaudio cudatoolkit=10.1 -c pytorch -y
@REM ) ELSE IF "%CUDA_VERSION%"=="cu102" (
@REM     call conda install pytorch==%TORCH_VERSION% torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
@REM ) ELSE IF "%CUDA_VERSION%"=="cu110" (
@REM     call conda install pytorch==%TORCH_VERSION% torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
@REM )

python -m pip install geomloss
@REM python -m pip install rdkit

REM PyTorch Geometric dependencies
@REM python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-scatter==2.0.9  -f https://data.pyg.org/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-sparse -f https://data.pyg.org/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-spline-conv -f https://data.pyg.org/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-geometric==%TORCH_GEOMETRIC_VERSION%
@REM python -m pip install torch-scatter==latest+%CUDA_VERSION% -f https://data.pyg.org/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-sparse==latest+%CUDA_VERSION%  -f https://data.pyg.org/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-cluster==latest+%CUDA_VERSION%  -f https://data.pyg.org/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-spline-conv==latest+%CUDA_VERSION% -f https://data.pyg.org/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-sparse==latest+%CUDA_VERSION%  -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-cluster==latest+%CUDA_VERSION%  -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-spline-conv==latest+%CUDA_VERSION% -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%.html
@REM python -m pip install torch-geometric==%TORCH_GEOMETRIC_VERSION%
@REM python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-%TORCH_VERSION%+%CUDA_VERSION%.html
@REM python -m pip install torch-geometric==%TORCH_GEOMETRIC_VERSION%
python -m pip install --no-cache-dir torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
python -m pip install --no-cache-dir torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
python -m pip install --no-cache-dir torch-cluster==1.5.8 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
python -m pip install --no-cache-dir torch-spline-conv==1.2.0 -f https://data.pyg.org/whl/torch-1.6.0+cpu.html
python -m pip install torch-geometric==1.6.0
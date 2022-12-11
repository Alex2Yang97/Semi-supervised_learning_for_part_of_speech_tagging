# ignore this file
cudaversion=${2:-"11.6"}

# for vscode config on remote
conda install -n jpmc-capstone ipykernel --update-deps --force-reinstall
# for script usage, install the related dependencies
conda install pandas 
conda install pytorch torchvision torchaudio pytorch-cuda=${cudaversion} -c pytorch -c nvidia
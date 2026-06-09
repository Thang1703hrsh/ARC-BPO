# 1) Download the right installer (auto-picks your OS/arch)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# 2) Install (interactive)
bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge3"

# 3) Initialize your shell, then restart terminal
"$HOME/miniforge3/bin/conda" init bash   # or zsh/fish etc.

# 4) Sanity check
conda --version
mamba --version

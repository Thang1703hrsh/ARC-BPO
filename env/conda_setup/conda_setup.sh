# Don’t auto-activate base
conda config --set auto_activate_base false

# Prefer conda-forge and make priorities strict
# conda config --add channels conda-forge
conda config --set channel_priority strict

# Ensure the fast solver (if needed on older conda)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

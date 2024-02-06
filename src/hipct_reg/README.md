# How To Check Registration

---

- Connect to `rnice`
- Open a terminal
- Ask for a Slurm job allocation for enough power to run the conversion:

```sh
salloc --x11 --partition=bm18 --exclusive --mem=0 --ntasks=1 --gres=gpu:2 --time=5:00:00 srun --pty bash
```

- Source the python environment to have all the correct library:

```sh
source /data/projects/hop/data_repository/Various/neuroglancer_pipeline/NP_env/bin/activate
```

- Run the checking python script:

```sh
python3 /data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/ITK_visu.py
```

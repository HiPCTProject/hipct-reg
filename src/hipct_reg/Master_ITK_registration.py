"""
This script launches multiple registration list files on the cluster as independent jobs.
"""

import glob
import os


def send_slurm(registration_list_file: str) -> None:
    """
    Parameters
    ----------
    registration_list_file : str
        Path to registration list .txt file.

    """
    print("Starting script")

    job_dir = "/data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/slurm/job/"
    output_dir = "/data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/slurm/output/"

    job_name = os.path.basename(registration_list_file)
    job_file = f"{os.path.dirname(job_dir)}/{job_name}.job"

    # -----------------------------------------------------------------
    # LINUX SCRIPT
    # -----------------------------------------------------------------
    sh_script = f"""#!/bin/bash
#SBATCH --output={output_dir}/slurm-%x-%j.%a.out
#SBATCH --partition=bm18
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
echo ------------------------------------------------------

echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST
echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR
echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST
echo SLURM_JOB_ID: $SLURM_JOB_ID
echo SLURM_JOB_NAME: $SLURM_JOB_NAME
echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION
echo SLURM_NTASKS: $SLURM_NTASKS
echo SLURM_CPUS-PER-TASK: $SLURM_CPUS_PER_TASK
echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo ------------------------------------------------------

echo Starting virtual environment
source /home/esrf/joseph08091994/python/pyEnv2/bin/activate
echo Virtual environment started

echo Starting registration script
srun python ITK_registration.py {registration_list_file}
"""
    # -----------------------------------------------------------------

    with open(job_file, "w") as f:
        f.write(sh_script)

    print(f"sbatch {job_file}")
    os.system(f"sbatch {job_file}")
    print("slurm sent")


if __name__ == "__main__":
    prefix = "registration_list_LADAF-2021-17_heart"

    registration_list = glob.glob(
        f"/data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/registration_list/{prefix}*.txt"
    )

    for line in registration_list:
        print(line)
        send_slurm(line)

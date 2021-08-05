#!/bin/bash

#SBATCH --account=def-sfabbro
#SBATCH --time=0-2:00

## create 100 jobs
#SBATCH --array=0-99
#SBATCH --mem=8000M
#SBATCH --output=outputs/%x-%j.out

# a small function that will call your python script to run on one tile
function infer_one_tile() {
    local tile=$1

    # where to download each tile
    local workdir=${SLURM_TMPDIR}/${tile}
    mkdir -p ${workdir}

    echo "downloading ${tile} files on host $(hostname)"
    date

    # PanSTARRS tiles (you may ignore this for now)
    #if [[ ${tile} =~ PS1 ]]; then
    #    vcp -L -v vos:cfis/ps_tiles/${tile}* ${workdir}/
    # CFIS tiles
    vcp -v vos:cfis/tiles_DR3/${tile}* ${workdir}/
    ls ${workdir}
    #fi

    echo "performing inference on ${tile}"
    date

    python inference_pipeline.py ${tile}

    #echo "saving outputs"
    #cp ${workdir} ${SCRATCH}/my_output.csv

    # cleanup, ready for next tile
    rm -fv ${workdir}
}


source $HOME/umap/bin/activate

# create a file 'tile.list' with list of all available tiles:
# 0. Get a certificate:

#   cadc-get-cert -u AnaHoban

# 1. List all files in the tiles_DR3 directory

#   vls vos:cfis/tiles_DR3 > tiles_DR3.vls

# 2. From the list of all files, create a file tile.list with unique tile names

#   cat tiles_DR3.vls | sed -e 's|\(CFIS........\).*|\1|g' | sort | uniq > tile.list


# create an array of all the tiles
tile_list=($(<tile.list))

# set the number of tiles that each SLURM task should do
per_task=1000

# starting and ending indices for this task
# based on the SLURM task and the number of tiles per task.
start_index=$(( (${SLURM_ARRAY_TASK_ID} - 1) * ${per_task} + 1 ))
end_index=$(( ${SLURM_ARRAY_TASK_ID} * ${per_task} ))

echo "This is task ${SLURM_ARRAY_TASK_ID}, which will do tiles ${tile_list[${start_index}]} to ${tile_list[${end_index}]}"

for (( idx=${start_index}; idx<=${end_index}; idx++ )); do
    tile=${tile_list[${idx}]}
    echo "This is SLURM task ${SLURM_ARRAY_TASK_ID} for tile ${tile}"
    infer_one_tile ${tile}
done
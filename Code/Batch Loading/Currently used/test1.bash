#!/bin/bash

#SBATCH --account=def-sfabbro
#SBATCH --time=0:15:00
#SBATCH --mem=8000M


source $HOME/umap/bin/activate

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
    #certificate /home/anahoban/.ssl/cadcproxy.pem
    vcp vos:cfis/tiles_DR3/${tile}* ${workdir}/
    
    for file in ${SLURM_TMPDIR}/*; 
    do
      echo "${file##*/}"
    done
    
    
    for file in ${workdir}/*; 
    do
      echo "${file##*/}"
    done
    #fi

    echo "performing inference on ${tile}"
    date

    python inference_pipeline.py ${tile} ${workdir}

    #echo "saving outputs"
    #cp ${workdir} ${SCRATCH}/my_output.csv

    # cleanup, ready for next tile
    rm -rf ${workdir}
}


# create an array of all the tiles
tile_list=($(<tile.list))

tile=${tile_list[0]}

infer_one_tile ${tile}

#!/bin/bash
#SBATCH --job-name=deep_gal         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=10          # nombre de coeurs à réserver (un quart du noeud)
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=2:00:00              # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=dg%j.out          # nom du fichier de sortie
#SBATCH --error=dg%j.out           # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_gpu-dev            # qos_gpu-t4 for long, qos_gpu-t3 for short, qos_gpu-dev for dev
#SBATCH --partition=gpu_p2

# nettoyage des modules charges en interactif et hérités par défaut
module purge
 
# chargement des modules
module load anaconda-py3/2019.03 cuda/10.0 cudnn/7.6.5.32-cuda-10.1 fftw/3.3.8 r
 
# echo des commandes lancées
set -x
 
# exécution du code
cd $WORK/repo/deep_galaxy_models
export PYTHONPATH=.
python scripts/mk_plots.py --generative_model=modules/flow_vae_maf_16/generator \
                           --out_dir=results \
			   --n_batches=20

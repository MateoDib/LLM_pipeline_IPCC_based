#!/bin/bash
#SBATCH -p publicgpu          # Partition publique avec des GPU
#SBATCH -N 1                  # 1 nœud
#SBATCH --exclusive           # Nœud entièrement dédié à ce job
#SBATCH --gres=gpu:2          # 2 GPU par nœud
#SBATCH --constraint=gpuh100  # Utiliser les GPU H100
#SBATCH --cpus-per-task=8     # Nombre de cœurs par tâche
#SBATCH --mem=500G            # Mémoire totale par nœud
#SBATCH -t 05:00:00           # Temps limite de 5 heures
#SBATCH --mail-type=BEGIN,END # Notifications e-mail début et fin
#SBATCH --mail-user=
#SBATCH -o output_h100.log    # Fichier de sortie pour les logs

# 1) Charger Python
module load python/python-3.11.4

# 2) Variables
TASK="12"   # Tâche à exécuter (1 à 12)

# 3) Activer votre env virtuel
source /home2020/home/beta/aebeling/python/bin/activate

echo "## CPU ##"
lscpu
echo "## RAM ##"
free -h
echo "## NVIDIA-SMI ##"
nvidia-smi

# 4) Lancement de main.py
python main_local.py \
    --task "${TASK}"

# 5) Désactivation env
deactivate
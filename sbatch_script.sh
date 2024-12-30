#!/bin/bash

#SBATCH --job-name=ram_gsam                 # sets the job name
#SBATCH --output=ram_gsam_output.txt 
#SBATCH --error=ram_gsam_error.txt                             # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=70:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=shenlong2                                     # set QOS, this will determine what resources can be requested
#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks-per-node=1                                              # request 4 cpu cores be reserved for your node total
#SBATCH --mem=64gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:1                                            # specify gpu usage, make sure your program is optimized to use this
#SBATCH --cpus-per-task=3                                      # number of cpu-cores per task

# module load anaconda/2022-May/3
eval "$(conda shell.bash hook)"
conda activate /projects/perception/personals/david/miniconda3_old/envs/gsam
module load cuda/.11.6

# python ram_gsam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --out_dir /projects/perception/datasets/4dunderstanding/data/sintel_val  --work_dir /projects/perception/datasets/4dunderstanding/data/sintel_val --ram_checkpoint ram_swin_large_14m.pth   --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image assets/demo9.jpg   --box_threshold 0.4   --text_threshold 0.4   --iou_threshold 0.5   --device "cuda"

# python ram_gsam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --work_dir /projects/perception/datasets/scannet200/ovir_preprocessed_data_val/scans --out_dir /projects/perception/datasets/scannet200/ovir_preprocessed_data_val/scans --ram_checkpoint ram_swin_large_14m.pth   --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image assets/demo9.jpg   --box_threshold 0.4   --text_threshold 0.4   --iou_threshold 0.5   --device "cuda"

# python ram_gsam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --work_dir /projects/perception/datasets/4dunderstanding/data/qualitative --ram_checkpoint ram_swin_large_14m.pth   --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image assets/demo9.jpg   --box_threshold 0.35   --text_threshold 0.35   --iou_threshold 0.5   --device "cuda"


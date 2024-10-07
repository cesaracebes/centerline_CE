
# Centerline-Cross Entropy loss

This repository contains the code necessary to replicate the experiments in our paper _The centerline-Cross Entropy loss for vessel-like structure segmentation: better topology consistency without sacrificing accuracy_. 

[poster](https://github.com/cesaracebes/centerline_CE/blob/main/MICCAI_2024_clCE.pdf)

## For the nnUNet-based coronary artery segmentation experiments:

### Code

Included in this repository are the files for the losses, the trainer classes, and the evaluation file that incorporates the clDice metric.

**To integrate this extension into nnUNet, you must add the files from this repository to the corresponding locations in your nnUNet installation as specified below:**

- For trainer classes and loss functions:
  - Place `nnUNetTrainerclDiceLoss.py` in `nnUNet/nnunet/training/nnUNetTrainer/` of your nnUNet installation.

- For loss definitions:
  - Place `cldice_loss.py` in `nnUNet/nnunet/training/loss/` of your nnUNet installation.

- For evaluation:
  - Place `evaluate_predictions.py` in `nnUNet/nnunet/evaluation/` of your nnUNet installation.

Please ensure that you follow the correct path structure within the nnUNet framework for seamless integration.


## For the retinal artery segmentation experiments:

### Code

Included in this repository is the code to directly replicate the experiments using the retinal dataset *FIVES*. To do so:
 - First, run `download_prepare_2d_data.sh`.
 - When the dataset is downloaded, run `retina_experiments_FIVES_x`, where "x" corresponds to the conditions in the FIVES dataset (FIVES-N for healthy eyes, FIVES-G for glaucomatous eyes, FIVES-A for age-related macular
degeneration and FIVES-D for diabetic retinopathy).


_Pending to expand repository information_

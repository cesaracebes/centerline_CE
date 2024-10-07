
set -e

# linux-specific for unraring
mkdir 7z
wget https://www.7-zip.org/a/7z2301-linux-x64.tar.xz
tar -xf 7z2301-linux-x64.tar.xz -C 7z
mv 7z/7zz .
rm -r 7z
rm 7z2301-linux-x64.tar.xz

mkdir -p data
nnUNet_raw=data # this should be your nnUnet_raw directory
## FIVES ###############################################################################################################
wget --no-check-certificate https://figshare.com/ndownloader/files/34969398
mkdir fives
./7zz x 34969398 -ofives
rm 34969398
python prepare_fives.py --nnUNet_raw $nnUNet_raw --source fives --dataset_name Dataset54_fives
rm -r fives
rm 7zz
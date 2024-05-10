# 1 arg = gcp link
gcp_link=$1

filename="mm_projector"

# 1. download the file
ckpt_path=./checkpoints/$filename
echo "Downloading from $gcp_link to $ckpt_path"
gcloud alpha storage rsync $gcp_link $ckpt_path
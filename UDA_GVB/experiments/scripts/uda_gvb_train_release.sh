#!/usr/bin/env bash
job_id=$1
config_file=$2


project_home='ghome/your_name/DA/UDA_test/'
export HOME='/ghome/your_name/DA/UDA_test'
export HOME=${HOME}
echo 'HOME is '${HOME}
cd ${HOME} || exit

trainer_class=gvb
validator_class=gvb
scripts_path=$HOME'/experiments/scripts/get_visible_card_num.py'
GPUS=$(python ${scripts_path})
PORT=16614

python_file=./train.py
# TODO: removing CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  ${python_file} --task_type cls --job_id ${job_id} --config ${config_file} \
  --trainer ${trainer_class} --validator ${validator_class}

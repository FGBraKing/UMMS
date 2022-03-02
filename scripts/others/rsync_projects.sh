#!/bin/bash
#set -ex
set -e
declare -A server0
declare -A server1
declare -A server2
declare -A server3

username='lf'

server0=(
  ["ip"]="172.21.141.4"
  ["home"]="/home/users/lf"
  ["working_dir"]="/home/users/lf/data_lf"
)
server1=(
  ["ip"]="172.21.141.5"
  ["home"]="/home/lf"
  ["working_dir"]="/home/lf/wind_lf"
)
server2=(
  ["ip"]="172.21.114.107"
  ["home"]="/home/lf"
  ["working_dir"]="/home/lf/data_fong"
)
server3=(
  ["ip"]="172.21.16.17"
  ["home"]="/home/lf"
  ["working_dir"]="/home/lf/raid_lf"
)

#echo "${server0[*]}"
#echo "${server3[@]}"
#echo "${!server3[*]}"

for id in $(seq 0 3)
do
#  eval  echo '$'"{server${id}[*]}"
  working_dir=$(eval echo '$'"{server${id}[working_dir]}")

  if [ -d "$working_dir" ]; then
    echo "base_dir: ${working_dir}"
    valid_id=${id}
    echo "valid_id: ${valid_id}"
  fi
done

source_data_dir=$(eval echo "\${server${valid_id}[working_dir]}/DATA/")
source_project_dir=$(eval echo "\${server${valid_id}[working_dir]}/PROJECT/")

echo "source_data_dir: ${source_data_dir}"
echo "source_project_dir: ${source_project_dir}"

# 字符串比较：!=
for id in $(seq 0 3)
do
#  echo "${id}"
  if [ "${id}" -ne "${valid_id}" ]; then
    target_ip=$(eval echo '$'"{server${id}[ip]}")
    target_working_dir=$(eval echo '$'"{server${id}[working_dir]}")
    target_data_dir="${username}@${target_ip}:${target_working_dir}/DATA"
    target_project_dir="${username}@${target_ip}:${target_working_dir}/PROJECT"
    echo "synchronizing from ${source_data_dir}  to  ${target_data_dir}"
    eval "rsync -avP --update --delete-after --exclude=".*" ${source_data_dir}  ${target_data_dir}"
    echo "synchronizing from ${source_project_dir}  to  ${target_project_dir}"
#    eval "rsync -avP --update --delete-after --exclude=".*" ${source_project_dir} ${target_project_dir}"
  fi
done
# --append --inplace
# --update  --size-only
# --existing --ignore-existing --delete-after
# --exclude="traces/*"
echo "*********************************************end*****************************************************************"
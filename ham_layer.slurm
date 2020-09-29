#!/bin/bash
#SBATCH -J HN-layer             # 作业名是 test
#SBATCH -p GPU               # 提交到 cpu 分区
#SBATCH -N 1                 # 使用一个节点
#SBATCH --cpus-per-task=4    # 每个进程占用一个 cpu 核心
#SBATCH -t 108:00:00          # 任务最大运行时间是 5 分钟
#SBATCH -o outputs/layer-0920.txt        # 将屏幕的输出结果保存到当前文件夹的 test.out
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1

# python ham_layer.py --layer 4
# python ham_layer.py --layer 10
# python ham_layer.py --layer 15
python ham_layer.py --layer 25
# python ham_layer.py --layer 30
# python ham_layer.py --layer 40
# python ham_layer.py --layer 50

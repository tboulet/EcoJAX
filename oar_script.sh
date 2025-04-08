#!/bin/bash 

#OAR -q production 
#OAR -l host=1/gpu=1
#OAR -l walltime=4:00:00
#OAR -p gpu-16GB AND gpu_compute_capability_major>=5
#OAR -O OAR_ej1.log
#OAR -E OAR_ej1.log



job_id=$OAR_JOB_ID
echo Running job $job_id on host $OAR_NODEFILE
hostname # display some information about attributed resources

## Setup
cd $HOME/projects/EcoJAX
source $HOME/venv/bin/activate

## Iterate over the seeds
initial_date=$(date +"%Y%m%d_%H%M%S")
seed_max=100
benchmark_name='bench_g'
model='region'
for k in $(seq 1 $seed_max); do
    seed=$RANDOM 
    list_omega=(0 0.1 0.2 0.6 1)
    for i in $(seq 0 $((${#list_omega[@]} - 1))); do
        omega=${list_omega[$i]}
        run_name="run_$benchmark_name-$initial_date-$model/omega_$omega/seed_$seed"
        log_dir="logs/$run_name"
        echo "Running $log_dir"
        mkdir -p "$log_dir"
        echo "Running python script with seed $seed and omega $omega"
        python run.py --config-name dgx do_wandb=True env/metrics=metrics_dgx +benchmark_name=$benchmark_name seed=$seed +run_name=\'$run_name\' \
            env.sum_energy_map_ref=275 \
            agents=ada_r_hc \
            env=fruits_plants \
            env.mode_variability_fruits=space_diffusion \
            env.omega=$omega \
            agents.do_include_id_fruit=True \
            model=$model \
            model.mlp_region_config.hidden_dims=[] \
            model.mlp_config.hidden_dims=[10] \
            > "$log_dir"/log.txt 2>&1
    done
done
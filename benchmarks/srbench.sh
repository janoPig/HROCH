# clone pmlb
git clone https://github.com/EpistasisLab/pmlb/
cd pmlb
git lfs pull
cd ..

# clone srbench
mkdir srbench
cd srbench
git clone https://github.com/janoPig/srbench.git
cd srbench
git checkout local_test

# install enviroment
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate srbench

cd experiment
# black-box experiment
python analyze.py ../../../pmlb/datasets -n_trials 10 -ml PHCRegressor -results ../results_blackbox -skip_tuning --noskips --local -n_jobs 10 -job_limit 100000

# submit the ground-truth dataset experiment. 
for data in "../../../pmlb/datasets/strogatz_" "../../../pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            $data"*" \
            -results ../results_sym_data \
            -ml PHCRegressor \
            --noskips \
            --local \
            -target_noise $TN \
            -sym_data \
            -n_trials 10 \
            -m 16384 \
            -job_limit 100000 \
            -skip_tuning \
            -n_jobs 10
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done

# assess the ground-truth models that were produced using sympy
for data in "../../../pmlb/datasets/strogatz_" "../../../pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            -script assess_symbolic_model \
            $data"*" \
            -results ../results_sym_data \
            -target_noise $TN \
            -ml PHCRegressor \
            --local \
            -sym_data \
            -n_trials 10 \
            -m 8192 \
            -time_limit 0:01 \
            -job_limit 100000 \
            -n_jobs 10
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done

# postprocessing
cd ../postprocessing
python collate_blackbox_results.py ../results_blackbox
python collate_groundtruth_results.py ../results_sym_data



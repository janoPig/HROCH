# SRBench

Benchmark [SRBench](https://github.com/cavalab/srbench)

Data [PMLB](https://github.com/EpistasisLab/pmlb)

## Testing

SRBench test 120 black-box and 133 ground-truth datasets(119 feynman + 14 strogatz). Every dataset is tested for 10 random seeds,
ground-truth datasets are testes for 4 values of noise[0, 0.001, 0.01, 0.1]. That's together (120+133*4)*10 = 6520 runs of symbolic regressor.

__On local computer__ this script [srbench.sh](srbench.sh) (requires miniconda)
Run time is reduced to 1 second per sample. Total 6520s / number of used cores + benchmark overhead.

__With github workflow__ [sr_bench.yml](../.github/workflows/sr_bench.yml)
Results are stored in bbox_result and gt_result artifacts. There are 4 choises HROCH_1s, HROCH_10s, HROCH_1m, HROCH_5m with defined time per sample to 1s, 10s, 1m, 5m. Total run time from 47 min to 17 hours.
  
## Benchmarking Results

### Results for Ground-truth Problems

![image](https://github.com/janoPig/HROCH/assets/75015989/3fa087dc-8caf-4301-86d7-4e79a4e84402)

#### Symbolically-verfied Solutions

How often a method finds a model symbolically equivalent to the ground-truth process

![image](https://github.com/janoPig/HROCH/assets/75015989/d36028fd-5d5c-4713-833c-a4999c15a7b2)

#### Accuracy Solutions

How often a method finds a model with test set R2>0.999

![image](https://github.com/janoPig/HROCH/assets/75015989/7c224295-f4e2-4c40-bb8b-a77c41442fb2)

### Results for Black-box Regression

![image](https://github.com/janoPig/HROCH/assets/75015989/6fd95437-e650-480e-b753-d4a4a52469d9)

#### Accuracy-Complexity Trade-offs

Considering the accuracy and simplicity of models simultaneously, this figure illustrates the trade-offs made by each method.
Methods lower and to the left produce models with better trade-offs between accuracy and simplicity.

![image](https://github.com/janoPig/HROCH/assets/75015989/4b529914-3c2e-4c64-be12-86478a556dd8)

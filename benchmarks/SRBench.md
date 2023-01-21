# SRBench

Benchmark [SRBench](https://github.com/cavalab/srbench)

Data [PMLB](https://github.com/EpistasisLab/pmlb)

## Testing

SRBench test 120 black-box and 133 ground-truth datasets(119 feynman + 14 strogatz). Every dataset is tested for 10 random seeds, 
ground-truth datasets are testes for 4 values of noise[0, 0.001, 0.01, 0.1]. That's together (120+133*4)*10 = 6520 runs of symbolic regressor. 

__On local computer__ this script [srbench.sh](srbench.sh) (requires miniconda)
Run time is reduced to 1 second per sample. Total 6520s / number of used cores + benchmark overhead.

__With github workflow__ [sr_bench.yml](../.github/workflows/sr_bench.yml)
Results are stored in bbox_result and gt_result artifacts. There are 4 choises PHCRegressor, PHCRegressor1, PHCRegressor2, PHCRegressor3 with defined time 
per sample to 1s, 10s, 1m, 5m. Total run time from 47 min to 17 hours. 
  

## Benchmarking Results


### Results for Ground-truth Problems

![image](https://user-images.githubusercontent.com/75015989/213884843-ff14dcb3-ecfd-4e03-b566-3629c465c971.png)


#### Effect of running time for HROCH

![image](https://user-images.githubusercontent.com/75015989/212563922-f6099e66-2865-4cab-84b1-a155eb4a6145.png)


#### Symbolically-verfied Solutions

How often a method finds a model symbolically equivalent to the ground-truth process

![image](https://user-images.githubusercontent.com/75015989/213884898-b0f27cde-64cd-4f4b-9b0d-86489425a05c.png)


#### Accuracy Solutions

How often a method finds a model with test set R2>0.999

![image](https://user-images.githubusercontent.com/75015989/213884914-d0f35304-8bc8-4b25-b399-9a40d007b053.png)


### Results for Black-box Regression

![image](https://user-images.githubusercontent.com/75015989/213884778-d4658242-9943-4c92-80fb-5a0397e5482b.png)


#### Accuracy-Complexity Trade-offs

Considering the accuracy and simplicity of models simultaneously, this figure illustrates the trade-offs made by each method. 
Methods lower and to the left produce models with better trade-offs between accuracy and simplicity. 

![image](https://user-images.githubusercontent.com/75015989/213884730-d4920bbd-9529-48c5-9c33-5061384160a1.png)


#### Compare with bbox regressors

![image](https://user-images.githubusercontent.com/75015989/213885323-a7902c5f-dcd3-4a39-a32c-7921e599324d.png)



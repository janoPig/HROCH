# SRBench

## Testing

SRBench test 120 black-box and 133 ground-truth datasets(119 feynman + 14 strogatz). Every dataset is tested for 10 random seeds, 
ground-truth datasets are testes for 4 values of noise[0, 0.001, 0.01, 0.1]. That's together (120+133*4)*10 = 6520 runs of symbolic regressor. 

__On local computer__ this script [srbench.sh](https://github.com/janoPig/HROCH/blob/a892dedb18b67491b7b4ebb3c704d6b092a424bd/benchmarks/srbench.sh) (requires miniconda)
Run time is reduced to 1 second per sample. Total 6520s / number of used cores + benchmark overhead.

__With github workflow__ [sr_bench.yml](https://github.com/janoPig/HROCH/blob/a892dedb18b67491b7b4ebb3c704d6b092a424bd/.github/workflows/sr_bench.yml)
Results are stored in bbox_result and gt_result artifacts. There are 4 choises PHCRegressor, PHCRegressor1, PHCRegressor2, PHCRegressor3 with defined time 
per sample to 1s, 10s, 1m, 5m. Total run time from 47 min to 17 hours. 
  

## Benchmarking Results


### Results for Ground-truth Problems

![image](https://user-images.githubusercontent.com/75015989/212563885-45126499-11f9-4af9-8502-003e5d7fbf33.png)

#### Effect of running time for HROCH

![image](https://user-images.githubusercontent.com/75015989/212563922-f6099e66-2865-4cab-84b1-a155eb4a6145.png)


#### Symbolically-verfied Solutions

How often a method finds a model symbolically equivalent to the ground-truth process

![image](https://user-images.githubusercontent.com/75015989/212563341-190f1746-1a29-49fe-8e5b-15dabfb555c9.png)


#### Accuracy Solutions

How often a method finds a model with test set R2>0.999

![image](https://user-images.githubusercontent.com/75015989/212563450-f7affb67-a5b5-44a1-9b49-656b581a4865.png)


### Results for Black-box Regression

![image](https://user-images.githubusercontent.com/75015989/212563809-a500e179-16a4-4e18-a2ab-1b7b697bb2b3.png)

#### Accuracy-Complexity Trade-offs

Considering the accuracy and simplicity of models simultaneously, this figure illustrates the trade-offs made by each method. 
Methods lower and to the left produce models with better trade-offs between accuracy and simplicity. 

![image](https://user-images.githubusercontent.com/75015989/212563824-3deb3ef8-33a0-411f-ab54-6391ce6687c3.png)

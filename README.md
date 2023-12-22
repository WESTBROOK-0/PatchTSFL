# PatchTSFL
# PatchTSFL framework
**Patching**：The Patching operation splits the long time series into multiple patches, and the number of patches is used as the input length of the encoder, which not only preserves local sequence information, but also reduces the complexity of the model. 

**Fourier enhanced block**：Using the Fourier enhanced block instead of the multi-headed attention mechanism in the traditional transformer. Important information in the time series is captured by converting the time domain data into a frequency domain mapping, while also reducing the complexity of the model. 

**MOEDecomp**：Using the Mixture Of Experts Decomposition block (MOEDecomp) to decompose the series enables the overall trend of the time series to be captured more fully.   
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/b86aae16-a470-4f1e-a850-576571b6cca2)

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/4fc6f94a-9d45-40ac-b2a0-16589bc589a4)

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/0f5d9c46-5256-4077-9d75-6ed58fbd56b9)

# Results
## Multivariate time series forecasting  

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/11f216f5-8412-4d7e-b5e4-09e2faddbc1d)


## Ablation Study

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/796b5f00-dec1-49f1-9e82-c22e6efdfe51)


## Effect of different input and prediction lengths on the model

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/b78043bf-9a24-42de-a7de-d09c399599c0)


## Visualization of model prediction results  

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/e92dce28-cd1b-4c31-a194-589e7cc936e3)

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/514b9054-8ee6-4ae3-9405-063a469fe2bf)  

## Efficiency Analysis  

![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/1b13bf96-7346-42f1-aeac-73fae9173941)


![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/7a7754e7-65c6-49e8-b9dd-64b386f68b1c)

## Unvariate time series forecasting


# Getting start  
1.Install requirements. pip install -r requirements.txt

2.Download data. You can download all the datasets from Autoformer. Create a seperate folder ./dataset and put all the csv files in the directory.

3.Training. All the scripts are in the directory ./scripts/PatchTST. The default model is PatchTSFL. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ./result.txt to see the results once the training is done: sh ./scripts/PatchTSFL/weather.sh

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.

# Acknowledgement  
We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/yuqinie98/PatchTST








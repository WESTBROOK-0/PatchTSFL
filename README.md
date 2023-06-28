# PatchTSFL
# PatchTSFL framework
**Patching**：The Patching operation splits the long time series into multiple patches, and the number of patches is used as the input length of the encoder, which not only preserves local sequence information, but also reduces the complexity of the model. 

**Fourier enhanced block**：Using the Fourier enhanced block instead of the multi-headed attention mechanism in the traditional transformer. Important information in the time series is captured by converting the time domain data into a frequency domain mapping, while also reducing the complexity of the model. 

**MOEDecomp**：Using the Mixture Of Experts Decomposition block (MOEDecomp) to decompose the series enables the overall trend of the time series to be captured more fully.   
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/f7989dbd-a268-4bf1-89fe-ab8615ac0de9)  
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/ca0a2ad7-fbd5-4a0d-820c-b3344119b5a7)  
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/e75f91fe-92d6-4107-b2fc-9d7d7c70a995)  

# Results

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








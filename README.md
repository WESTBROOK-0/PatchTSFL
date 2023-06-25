# PatchTSFL
# PatchTSFL framework
**Patching**：The Patching operation splits the long time series into multiple patches, and the number of patches is used as the input length of the encoder, which not only preserves local sequence information, but also reduces the complexity of the model.    
**Fourier enhanced block**：Using the Fourier enhanced block instead of the multi-headed attention mechanism in the traditional transformer. Important information in the time series is captured by converting the time domain data into a frequency domain mapping, while also reducing the complexity of the model.    
**MOEDecomp**：Using the Mixture Of Experts Decomposition block (MOEDecomp) to decompose the series enables the overall trend of the time series to be captured more fully.   
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/f7989dbd-a268-4bf1-89fe-ab8615ac0de9)  
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/ca0a2ad7-fbd5-4a0d-820c-b3344119b5a7)  
![image](https://github.com/WESTBROOK-0/PatchTSFL/assets/59240114/e75f91fe-92d6-4107-b2fc-9d7d7c70a995)  

# Results






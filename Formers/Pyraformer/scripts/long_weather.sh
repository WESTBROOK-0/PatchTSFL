# cd Pyraformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


python long_range_main.py  -data_path weather.csv -data weather \
-input_size 336 -predict_step 96 -n_head 6 -lr 0.00001 -d_model 256 >./logs/LongForecasting/Pyraformer_weather336_96.log
python long_range_main.py  -data_path weather.csv -data weather \
-input_size 336 -predict_step 192 -n_head 6 -lr 0.00001 -d_model 256 >./logs/LongForecasting/Pyraformer_weather336_192.log
# ALL scripts in this file come from Autoformer
if [ ! -d "./Informerlogs" ]; then
    mkdir ./Informerlogs
fi

if [ ! -d "./Informerlogs/LongForecasting" ]; then
    mkdir ./Informerlogs/LongForecasting
fi

random_seed=2021
model_name=Informer

for pred_len in 96 192
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_336_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 336 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 >Informerlogs/LongForecasting/$model_name'_weather336_'$pred_len.log
done
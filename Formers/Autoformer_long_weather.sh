# ALL scripts in this file come from Autoformer
if [ ! -d "./Autoformerlogs" ]; then
    mkdir ./Autoformerlogs
fi

if [ ! -d "./Autoformerlogs/LongForecasting" ]; then
    mkdir ./Autoformerlogs/LongForecasting
fi

random_seed=2021
model_name=Autoformer

for pred_len in 720
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 336 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --itr 1  >Autoformerlogs/LongForecasting/$model_name'_electricity336_'$pred_len.log
done
# ALL scripts in this file come from Autoformer
if [ ! -d "./Informerlogs" ]; then
    mkdir ./Informerlogs
fi

if [ ! -d "./Informerlogs/LongForecasting" ]; then
    mkdir ./Informerlogs/LongForecasting
fi

random_seed=2021
model_name=Informer

for pred_len in 720
do
  python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --patience 20\
      --itr 1  >Informerlogs/LongForecasting/$model_name'_Ettm2_'$pred_len.log
done

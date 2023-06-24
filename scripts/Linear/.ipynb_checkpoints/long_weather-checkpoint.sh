# add --individual for DLinear-I
if [ ! -d "./DLinearlogs" ]; then
    mkdir ./DLinearlogs
fi

if [ ! -d "./DLinearlogs/LongForecasting" ]; then
    mkdir ./DLinearlogs/LongForecasting
fi
seq_len=336
model_name=DLinear


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'1440 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 1440 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >DLinearlogs/LongForecasting/$model_name'_'Weather_$seq_len'_'1440.log

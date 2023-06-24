if [ ! -d "./FEDformerlogs" ]; then
    mkdir ./FEDformerlogs
fi

if [ ! -d "./FEDformerlogs/LongForecasting" ]; then
    mkdir ./FEDformerlogs/LongForecasting
fi
if [ ! -d "./FEDformerlogs/LongForecasting/univariate" ]; then
    mkdir ./FEDformerlogs/LongForecasting/univariate
fi

for preLen in 96 192 336 720
do

python -u run.py \
  --is_training 1 \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model FEDformer \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >./FEDformerlogs/LongForecasting/FEDformer_ETTh1_$pred_len.log
  
done
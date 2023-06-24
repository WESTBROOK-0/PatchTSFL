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

# ETTm1
python -u run.py \
  --is_training 1 \
  --data_path ETTm1.csv \
  --task_id ETTm1 \
  --model FEDformer \
  --data ETTm1 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1  >./FEDformerlogs/LongForecasting/univariate/FEDformer_ETTm1_$preLen.log

# ETTh1
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
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >./FEDformerlogs/LongForecasting/univariate/FEDformer_ETTh1_$preLen.log

# ETTm2
python -u run.py \
  --is_training 1 \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model FEDformer \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >./FEDformerlogs/LongForecasting/univariate/FEDformer_ETTm2_$preLen.log

# ETTh2
python -u run.py \
  --is_training 1 \
  --data_path ETTh2.csv \
  --task_id ETTh2 \
  --model FEDformer \
  --data ETTh2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 >./FEDformerlogs/LongForecasting/univariate/FEDformer_ETTh2_$preLen.log
  
done
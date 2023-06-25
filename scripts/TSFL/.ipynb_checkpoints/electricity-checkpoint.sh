if [ ! -d "./TSFLlogs" ]; then
    mkdir ./TSFLlogs
fi

if [ ! -d "./TSFLlogs/LongForecasting" ]; then
    mkdir ./TSFLlogs/LongForecasting
fi
seq_len=336
model_name=TSFL

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id Electricity_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 4 --learning_rate 0.0001 >TSFLlogs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
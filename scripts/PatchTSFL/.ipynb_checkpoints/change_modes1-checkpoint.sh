if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/change_modes" ]; then
    mkdir ./logs/LongForecasting/change_modes
fi
seq_len=336
model_name=PatchTSFL

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

for modes in 2 4 8 16 32 64 128 256
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id weather_$seq_len'low_'$modes \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len 96 \
      --mode_select 'low' \
      --modes $modes \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/change_modes/$model_name'_'$model_id_name'_'$seq_len'low_'$modes.log
done
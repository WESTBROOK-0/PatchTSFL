if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchTSFL

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
for pred_len in 336 720 1440
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
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
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


if [ ! -d "./DLinearlogs" ]; then
    mkdir ./DLinearlogs
fi

if [ ! -d "./DLinearlogs/LongForecasting" ]; then
    mkdir ./DLinearlogs/LongForecasting
fi

for pred_len in 336 720 1440
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >DLinearlogs/LongForecasting/DLinear'_'Weather_$seq_len'_'$pred_len.log
done

if [ ! -d "./Autoformerlogs" ]; then
    mkdir ./Autoformerlogs
fi

if [ ! -d "./Autoformerlogs/LongForecasting" ]; then
    mkdir ./Autoformerlogs/LongForecasting
fi

 python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_1440 \
    --model Autoformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 1440 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 >Autoformerlogs/LongForecasting/Autoformer'_weather96_'1440.log

if [ ! -d "./Informerlogs" ]; then
    mkdir ./Informerlogs
fi

if [ ! -d "./Informerlogs/LongForecasting" ]; then
    mkdir ./Informerlogs/LongForecasting
fi

 python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_1440 \
    --model Informer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 1440 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 2 >Informerlogs/LongForecasting/Informer'_weather96_'1440.log

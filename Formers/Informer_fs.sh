if [ ! -d "./Informerlogs" ]; then
    mkdir ./Informerlogs
fi

if [ ! -d "./Informerlogs/LongForecasting" ]; then
    mkdir ./Informerlogs/LongForecasting
fi

if [ ! -d "./Informerlogs/LongForecasting/univariate" ]; then
    mkdir ./Informerlogs/LongForecasting/univariate
fi

random_seed=2021
model_name=Informer

for pred_len in 96 192 336 720
do

  python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features S \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1  >Informerlogs/LongForecasting/univariate/$model_name'_fts_Etth1_'$pred_len.log
done

for pred_len in 96 192 336 720
do

  python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features S \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1  >Informerlogs/LongForecasting/univariate/$model_name'_fts_Etth2_'$pred_len.log
done

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_96_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features S \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1  >Informerlogs/LongForecasting/univariate/$model_name'_fts_Ettm1_'$pred_len.log
done

for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features S \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --itr 1  >Informerlogs/LongForecasting/univariate/$model_name'_fts_Ettm2_'$pred_len.log
done

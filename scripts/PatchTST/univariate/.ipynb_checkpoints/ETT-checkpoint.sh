if [ ! -d "./PatchTSTlogs" ]; then
    mkdir ./PatchTSTlogs
fi

if [ ! -d "./PatchTSTlogs/LongForecasting" ]; then
    mkdir ./PatchTSTlogs/LongForecasting
fi

if [ ! -d "./PatchTSTlogs/LongForecasting/univariate" ]; then
    mkdir ./PatchTSTlogs/LongForecasting/univariate
fi

seq_len=336
model_name=PatchTST
random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >PatchTSTlogs/LongForecasting/univariate/$model_name'_fS_'ETTh1'_'$seq_len'_'$pred_len.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >PatchTSTlogs/LongForecasting/univariate/$model_name'_fS_'ETTh2'_'$seq_len'_'$pred_len.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
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
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >PatchTSTlogs/LongForecasting/univariate/$model_name'_fS_'ETTm1'_'$seq_len'_'$pred_len.log
done

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
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
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >PatchTSTlogs/LongForecasting/univariate/$model_name'_fS_'ETTm2'_'$seq_len'_'$pred_len.log
done



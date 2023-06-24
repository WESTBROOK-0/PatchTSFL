# cd FEDformer
if [ ! -d "./FEDformerlogs" ]; then
    mkdir ./FEDformerlogs
fi

if [ ! -d "./FEDformerlogs/LongForecasting" ]; then
    mkdir ./FEDformerlogs/LongForecasting
fi

for preLen in 720
do
# weather
python -u run.py \
 --is_training 1 \
 --data_path electricity.csv \
 --task_id ECL \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 336 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 1 >./FEDformerlogs/LongForecasting/FEDformer_electricity336_$preLen.log
done
使用时要放到根目录下

nohup bash -c '
for i in 1 2 3 4 5; do
    echo "=== 实验 $i/5 开始 ==="
    python main.py --dataset omim-ordo --mode full --run_id aug_run_omim-ordo${i}
    echo "=== 实验 $i/5 完成 ==="
done
' > logs/aug_run_omim-ordo.log 2>&1 &

ps aux | grep analyze_propagation.py
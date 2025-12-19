# run sem

for KERNEL_TYPE in `seq 0 1`; do
    time python step0_database.py model.txt 0.01 1 500 $KERNEL_TYPE
    python step1_benchmark.py 19  $KERNEL_TYPE
done
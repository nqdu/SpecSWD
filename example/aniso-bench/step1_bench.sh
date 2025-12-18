#!/bin/bash
set -e 

# clean directory
rm -rf out 
mkdir -p out 

sourcedir=../../

# run sem
# for i in `seq 0 99`;
# do 
#     phi=`echo $i | awk '{print 0 + 360 / 99 * $1}'`
#     $sourcedir/bin/surftti modelhti.txt $phi 0.01 0.01 1

#     mv out/swd.txt out/swd.$i.txt 
# done
$sourcedir/bin/surfani model.txt 0.01 1 100 18 0


# bin2h5
echo "converting to hdf5..."
python $sourcedir/scripts/binary2h5.py out/database.bin out/swd.txt out/kernels.h5


# plot eigenfunctions
for mode in `seq 0 4`;
do 
    python plot_kernels.py out/kernels.h5 70 $mode
done

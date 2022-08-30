
#!/bin/csh
# c-shell script for convolutional neural network.
setenv HHH '/home/ysjoo/data_assimilation'   # path of this package


@ gpu_number = 0

setenv EXP 'DeepDA'



mkdir -p {$HHH}/output/$EXP/src

cd {$HHH}/output/$EXP/src
cp -f {$HHH}/code_paper/sample/obs.evluate.sample .



@ ENS = 1
while( $ENS <= 10)



# Run Training
sed "s#homedir#$HHH#g"            obs.evluate.sample > tmp10
sed "s/eexxpp/$EXP/g"             tmp10 > tmp20
sed "s/gpu_number/$gpu_number/g"  tmp20 > tmp10
sed "s/ensemble/$ENS/g"           tmp10 > tmp20
sed "s/totalepoch/$EEE/g"         tmp20 > obs.evluate.py

python obs.evluate.py


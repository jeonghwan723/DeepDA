
#!/bin/csh
# c-shell script for convolutional neural network.
setenv HHH '/home/ysjoo/data_assimilation'   # path of this package


@ gpu_number = 0

setenv EXP 'DeepDA'

setenv Epoch 100

mkdir -p {$HHH}/output/$EXP/src

cd {$HHH}/output/$EXP/src
cp -f {$HHH}/code_paper/sample/run.train.sample .
cp -f {$HHH}/code_paper/src/*.py .


# Run Training
sed "s#homedir#$HHH#g"            run.train.sample > tmp10
sed "s/eexxpp/$EXP/g"             tmp10 > tmp20
sed "s/gpu_number/$gpu_number/g"  tmp20 > tmp10
sed "s/epoch/$Epoch/g"         tmp10 > run.train.py

python run.train.py





cp -f {$HHH}/code_paper/sample/mail.sample .

sed "s/expname/$EXP/g"    mail.sample > mail.py

python mail.py




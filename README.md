# m3dv
EE228 课程大作业
代码说明：
./dataset：样本数据集
./mylib：模型需要的一些函数，其中包括模型，这一部分代码使用dudcheng/DenseSharp/mylib/models和dudcheng/DenseSharp/mylib/utils中的部分文件
./tmp：初始权重文件和训练时保存的模型
dataloader.py：下载数据集的函数，参考dudcheng/DenseSharp/mylib/dataloader进行了整合
Submission.csv：测试集预测文件，用于Kaggle提交
test.py：预测测试集样本，生成Submission.csv
train.py：训练模型函数

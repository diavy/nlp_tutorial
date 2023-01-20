### 任务四：基于LSTM+CRF的序列标注

https://github.com/Alic-yuan/nlp-beginner-finish/tree/master/task4
用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

## 参考

   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf

## 运行

训练：python main.py<br />

源代码只能跑通NERLSTM_CRF，无法跑通NERLSTM
经过修改后，NERLSTM和NERLSTM_CRF都可以训练了（只需要修改config中的模型名称即可），predict部分还需要修改（已修改跑通）

在执行前需要注意安装torchcrf的方法
pip install pytorch-crf

执行以下命令完成训练
nohup python -u main.py >> NERLSTM.log 2>&1 &
修改config中的模型名称为NERLSTM_CRF
nohup python -u main.py >> NERLSTM_CRF.log 2>&1 &
两个模型在该数据集上效果基本一样。

测试
nohup python -u main.py >> NERLSTM_CRF_predict.log 2>&1 &

ns表示的是地名，nr表示的是人名，nt表示的是机构团体

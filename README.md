# air_pollution 
 
去掉deep限制，MSE就变成91，98了。   
 
 
![完整的图](pic/20180304_0.png)   
这是去掉最不相关的的weather、wind做出来的图，
use_min_max_scaler = True
use_all_data = False
use_CCA_data = False
use_pm25_history = True
use_drop_least_importance = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122，MSE变为了73和87，比使用全部数据的要差一些。   
 
  
![完整的图](pic/20180303_3.png)   
这是使用cca算出来的最相关的pm10，co，temperature，moisture， pressure做出来的图，
use_min_max_scaler = True
use_all_data = False
use_CCA_data = True
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122，MSE变为了101和123，完全没有使用全部数据的好。  
 
 
![完整的图](pic/20180303_1.png)  ![完整的图](pic/20180303_2.png)   
这是使用全部数据做出来的图，
use_min_max_scaler = True
use_all_data = True
use_CCA_data = False
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122，貌似是现在最好的了。  
 
 
![完整的图](pic/20180303_0.png)  
ic/20180303_0.png) 这是使用randomforest算出来的最相关特征pm10和co做出来的图，
use_min_max_scaler = True
use_all_data = False  # have not completed
use_CCA_data = True
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122，看起来DBN效果比之前用temperature和moisture特征做的好，
待会再加入第三相关特征试试。 

![完整的图](pic/data_importance_0.png)![完整的图](pic/data_importance_1.png)  
可以看出与PM2.5最强相关的是pm10 
![完整的图](pic/data_importance_2.png)  
与PM2.5次相关的是co 


 
 
---  
以下的是使用temperature和moisture特征的，
并不是最新的用全部数据算出的最相关特征
![完整的图](pic/20180302_1.png)  
~~use_min_max_scaler = True
use_all_data = False
use_CCA_data = True
use_pm25_history = False
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122的完整图，
可以看出去掉PM2.5历史数据之后，图像变得非常糟糕，
可以说用其他的数据可以学习出PM2.5的变化趋势，
但是具体的PM2.5的值，还是要靠PM2.5的历史数据才能推测出。~~ 


![完整的图](pic/20180302_0.png)  
~~use_min_max_scaler = True
use_all_data = False
use_CCA_data = True
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122的完整图，可以看出
预测的值跟前一个实际值强相关，应该是过拟合了。~~ 










--- 

## 3月3日

### 工作概要


1. 700个数据，前400个数据为训练集，后500个数据作为测试集，并加入了一些气象特征，预测空气污染指数PM2.5，预测效果看起来
不错。avg_DBN(平均误差)=0.273 

![完整的图](pic/图片1.png)   
2. 改成在线式的预测，每当读入新的数据就继续预测，边训练边预测，从第50个数据开始训练，第51个数据开始预测，预测步长step=1，每次训练都是拿最近的50组数据进行训练。avg_DBN(平均误差)=0.131

![完整的图](pic/图片2.png)  
3.  后来发现预测的时候训练和预测的顺序反了，相当于事先就将当前的数据样本包含最新的数据先训练了一遍，然后在去预测当前值，效果当然好啦。于是改变了顺序，第一张图是纠正之后的结果，发现avg_DBN(平均误差)=0.302。单独用Pm2.5预测，得到第一幅图，接着有把pm2.5从数据集中剔除掉，只留下现在训练集中只留下气象特征，预测Pm2.5，得到第二幅图，可以发现，这些气象数据与pm2.5相关度太低，考虑到数据集的准确度问题，于是决定换新数据集。

![完整的图](pic/图片3.png)  

![完整的图](pic/图片4.png)  
4. 找到了新的空气和气象数据网站，能够收集每小时的数据。
空气质量数据网站：https://www.aqistudy.cn/
气象数据网站：http://data.cma.cn/data/detail/dataCode/A.0012.0001.html
云盘里有人每周更新全国各站点的每小时空气监测数据
https://pan.baidu.com/s/1gd8GUxt#list/path=%2F%E5%85%AC%E5%BC%80%2F%E5%85%A8%E5%9B%BD%E7%A9%BA%E6%B0%94%E8%B4%A8%E9%87%8F&parentPath=%2F%E5%85%AC%E5%BC%80
5.  在新的数据集里重新实验，从第50个数据开始训练，第51个数据开始预测，预测的步长是1。并用adaboost做对比，adaboost是一种迭代算法，其核心思想是针对同一个训练集训练不同的分类器(弱分类器)，然后把这些弱分类器集合起来，构成一个更强的最终分类器（强分类器），发现DBN的效果比原先旧的数据集效果要好了，但是比adaboost差一点。
use_min_max_scaler = True
use_all_data = True
step = 1
train_deep = 50
train_start = 50
predict_start = 51

![完整的图](pic/图片5.png)  

_avg_DBN:0.247299   loss_avg_AdaBoost:0.182487
DBN:	R-squared: 0.615542
MSE: 211.726538
INFO:root:AdaBoost:	R-squared: 0.806699
MSE: 106.453403
6.  于是，就用dbn替换了adaboost中的弱分类器，加了svm对比，用sigmoid代替relu激活函数。因为一个非常大的梯度经过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了。如果这种情况发生，那么从此所有流过这个神经元的梯度将都变成 0 。也就是说，这个 ReLU 单元在训练中将不可逆转的死亡，导致了数据多样化的丢失。从图中可以发现，使用sigmoid函数之后dbn-adaboost预测的效果有了一点提高，而且dbn-adaboost的抖动较小。

![完整的图](pic/图片6.png)  

use_min_max_scaler = True
use_all_data = True
step = 1
train_deep = 50
train_start = 50
predict_start = 51

l_avg_D:0.201764    l_avg_DA:0.177978   l_avg_S:0.589312
DBN:	R-squared: 0.810164
MSE: 104.545341
INFO:root:DBNAdaBoost:	R-squared: 0.794436
MSE: 113.207110

将DBN-AdaBoost与AdaBoost对比发现他们的预测效果很接近，但是比单纯的DBN准确率要高。

![完整的图](pic/图片7.png)  

参数设置

use_min_max_scaler = True
use_all_data = True
step = 1
train_deep = 50
train_start = 50
predict_start = 51

平均误差

l_avg_D:0.201259    l_avg_DA:0.188396  l_avg_A:0.185118
DBN:	R-squared: 0.810932
MSE: 104.122564
INFO:root:DBNAdaBoost:	R-squared: 0.759444
MSE: 132.477562
INFO:root:AdaBoost:	R-squared: 0.808426
MSE: 105.502396
7.  使用CCA和randomforest（随机森林）对PM2.5与温度（temperature）、风力 （wind）、 天气（weather）、湿度（ moisture）作相关度分析，发现温度和湿度的相关相关性更大，于是就将保留温湿度特征的数据在算法中实现。
![完整的图](pic/图片8.png)  

将预测深度deep调到了120，DBN效果最好，DBN-AdaBoost的准确率也提高了。

![完整的图](pic/图片9.png)  

参数设置

use_min_max_scaler = True
use_all_data = False
use_CCA_data = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122

平均误差

l_avg_D:0.138578    l_avg_DA:0.160282  l_avg_A:0.167904   l_avg_S:0.196340
DBN:	R-squared: 0.666527
MSE: 100.855421
INFO:root:DBNAdaBoost:	R-squared: 0.629013
MSE: 112.201321
INFO:root:AdaBoost:	R-squared: 0.585469
MSE: 125.370641
8.  考虑到PM2.5可能与其历史数据强相关，尝试将PM2.5从data里去掉，其它的气象特征能预测PM2.5的趋势，但结果不够准确，因此具体的PM2.5的值，还是要靠PM2.5的历史数据才能推测出。

![完整的图](pic/图片10.png)  

于是就考虑pm2.5与PM10、CO、SO2等指标一起预测PM2.5。经过CCA和randomforst相关的分析得到PM10、CO与PM2.5强相关，所以可以放到data里进行预测，准确率比之前加气象参数的又提高了，虽然图像看起来不好拟合，但是可以考虑继续加相关的特征。

![完整的图](pic/图片11.png)  

use_min_max_scaler = True
use_all_data = False  # have not completed
use_CCA_data = True
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122

l_avg_D:0.119074    l_avg_A:0.133971   l_avg_S:0.138150
DBN:	R-squared: 0.733514
MSE: 80.596038
INFO:root:AdaBoost:	R-squared: 0.662733
MSE: 102.003027

用全部数据跑了一遍，是目前最准确的。

![完整的图](pic/图片13.png)  

![完整的图](pic/图片12.png)  

use_min_max_scaler = True
use_all_data = True
use_CCA_data = False
use_pm25_history = True
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122


l_avg_D:0.120663	l_avg_A:0.129982	l_avg_S:0.217349
DBN:	R-squared: 0.762488
MSE: 71.845210
INFO:root:AdaBoost:	R-squared: 0.740108
MSE: 78.601664
* 啦啦 


> 我是引用 



`
I 
am 
code
`


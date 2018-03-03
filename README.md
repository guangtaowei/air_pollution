# air_pollution 
 
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
这是使用randomforest算出来的最相关特征pm10和co做出来的图，
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

## 我是二级标题 

### 我是三级 

我是正文 

我是第二段 

1. 第一点 
2. 第二点 

* 啦啦 
* 会更好 

+ ff 
+ jj 

- gfdg 
- hdfh 

> 我是引用 



`
I 
am 
code
`


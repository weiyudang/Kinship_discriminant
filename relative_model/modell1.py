#encoding:utf-8
#by yudang.wei
import  pandas as pd
from utils import   *
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, auc,roc_auc_score,precision_score,accuracy_score,f1_score,recall_score,auc

##1.数据清理
rel_data=pd.read_csv('../data/relatives_c2c.csv',index_col=0)
rel_data.drop(['custorm_id.1','number.1','custorm_id.2','number.2','custorm_id.3','number.3',],axis=1,inplace=True)
rel_data.fillna(0,inplace=True)

## 2.打标签
rel_data['is_rel']=get_relative(rel_data.name)
rel_df=rel_data[rel_data.is_rel==1]
unrel_df=rel_data[rel_data.is_rel==0]
rel_num=rel_df.shape[0]
unrel_num=unrel_df.shape[0]

##3. 采样：数据不平衡的情况下尽量使用重采样

## 下采样（Undersampling）:对非亲属数据欠采样，使得数据保持均衡
# sampler = np.random.permutation(unrel_num)[:rel_num*3]
# unrel_df=rel_data[rel_data.is_rel==0].take(sampler)

### 上采样（Oversampling）：对亲属数据重采样，保持数据均衡
sampler = np.random.randint(0,rel_num,size=unrel_num)
rel_df=rel_data[rel_data.is_rel==1].take(sampler)
rel_data=pd.concat([rel_df,unrel_df],axis=0)
rel_data.drop(['name','number'],axis=1,inplace=True)
### smote算法  python


## 4 测试集、训练集合分开
rel_feature=rel_data.ix[:,:-1]
target=rel_data.is_rel
X_train, X_test, y_train, y_test = train_test_split(rel_feature, target)

## 模型训练single
params = {
    'max_depth': 6,
    'n_estimators': 600,
    'subsample': 0.95,
    'colsample_bytree': 0.3,
    'learning_rate': 0.05,
    'reg_alpha': 0.1
}
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model = xgb.XGBClassifier()
X_train, X_test, y_train, y_test = train_test_split(rel_feature, target)
print roc_auc_score(y_test, y_pred)
print precision_score(y_test,y_pred)
print accuracy_score(y_test,y_pred)
print f1_score(y_test,y_pred)
print recall_score(y_test,y_pred)


### 参数变化
metrics=[]
metrics_name=['auc','precison','accuracy','f1','recall']
for max_depth in [2, 5, 10]:
    for n_estimators in [100, 200, 300]:
        params = {'max_depth': max_depth, 'n_estimators': n_estimators}

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics.append([(roc_auc_score(y_test, y_pred) ),precision_score(y_test,y_pred),accuracy_score(y_test,y_pred),\
      f1_score(y_test,y_pred),recall_score(y_test,y_pred)])
metrics_df=pd.DataFrame(np.array(metrics),columns=metrics_name)


##  CV 交叉验证
params1={
'booster':'gbtree',
'objective': 'binary:logistic',
'scale_pos_weight': 1/7.5,
#7183正样本
#55596条总样本
#差不多1:7.7这样子
'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':8, # 构建树的深度，越大越容易过拟合
'lambda':3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
#'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.03, # 如同学习率
'seed':1000,
'nthread':12,# cpu 线程数
'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 5000 # 迭代次数

X_train, X_test, y_train, y_test = train_test_split(rel_feature, target)
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_test, y_test)


# return 训练和验证的错误率
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, dtrain,num_boost_round=7000,evals=watchlist,early_stopping_rounds=500)

print ("跑到这里了save_model")
model.save_model('../model/20170201_B.model') # 用于存储训练出的模型
print ("best best_ntree_limit",model.best_ntree_limit)   #did not save the best,why?
print ("best best_iteration",model.best_iteration) #get it?

preds_x = model.predict(X_test,ntree_limit=model.best_iteration)#
print ks(preds_x,y_test)
##  参数调优化






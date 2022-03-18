import pandas as pd

data = pd.read_csv('../data/deal_data_hlx.csv')

data1 = data[:2190].round(5)
data2 = data[2190:].round(5)
data1.to_csv('../data/deal_data_hlx_train.csv', index=0)
data2.to_csv('../data/deal_data_hlx_test.csv', index=0)
print('ss')
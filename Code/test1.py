import cv2
import pandas as pd

dataset = pd.read_excel('../data/initial_hlx.xlsx', header=1, index_col=1, parse_dates=True)

indexs = pd.date_range('20190101', '20200430', freq='4H').format()
res = []
for item in indexs:
    item = item[:-3]
    if item in dataset.index:
        tmp = dataset.loc[item]
    else:
        tmp = pd.Series(index=dataset.columns)
    res.append(tmp)

dataf = pd.DataFrame(res, index=indexs)
deal_data = dataf.loc[:, ['pH(无量纲)', '溶解氧(mg/L)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)', '总磷(mg/L)', '总氮(mg/L)']].fillna(
    method='ffill')
deal_data.to_csv('../data/deal_data_hlx.csv')

print('ss')

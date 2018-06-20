import pandas as pd

df = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/portal-GeCKO-2018-05-30.csv', header=0, index_col=0, na_values='NaN')
X = (df - df.min()) / (df.max() - df.min())
X.to_csv('C:/Users/Vaibhav/nmf/inputs/portal-GeCKO-2018-05-30-n.csv')
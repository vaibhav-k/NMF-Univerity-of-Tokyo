import pandas as pd

df = pd.read_csv(r'C:/Users/Vaibhav/nmf/inputs/portal-Avana-2018-06-08.csv', header=0, index_col=0, na_values='NaN')
X = (df - df.min()) / (df.max() - df.min())
X.to_csv('C:/Users/Vaibhav/nmf/inputs/portal-Avana-2018-06-08-n.csv')
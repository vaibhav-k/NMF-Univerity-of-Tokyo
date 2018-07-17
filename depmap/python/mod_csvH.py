import pandas as pd

X = pd.read_csv(r'../H50.csv', index_col=0, header=0, na_values='NaN')
newhead = []
for word in list(X):
	word = word.split('_',1)[-1]
	newhead.append(word)
Y = pd.DataFrame(X.values, index=list(X.index), columns=newhead)
Y.to_csv('../H50mod.csv')
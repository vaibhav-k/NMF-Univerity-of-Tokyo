import pandas as pd
import csv 

x = pd.read_csv(r'../W50.csv', index_col=0, header=0, na_values='NaN')
x=x.T
clusters = {}
for r in x.index:
	clusters[r] = []
	for c in x.columns:
		if x[c][r]==1:
			clusters[r].append(c.split('.',1)[0])

for key, val in clusters.items():
	genes = clusters.get(key)
	with open("../matrices/rank=50,cluster=%d.txt" % int(key),'w') as resultFile:
		#wr = csv.writer(resultFile, dialect='excel')
		#wr.writerow(genes)
		for item in genes:
  			resultFile.write("%s\n" % item)

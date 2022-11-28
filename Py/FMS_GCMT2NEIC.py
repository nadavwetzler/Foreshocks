import pandas as pd
import numpy as np
f_neic = '/Users/nadavwetzler/Library/CloudStorage/GoogleDrive-nadav.wetzler@gmail.com/My Drive/DataSet/UCSC/Foreshocks/Catalogs_csv/NEIC.csv'
f_gcmt = '/Users/nadavwetzler/Library/CloudStorage/GoogleDrive-nadav.wetzler@gmail.com/My Drive/DataSet/UCSC/Foreshocks/Catalogs_csv/GCMT2.csv'
neic = pd.read_csv(f_neic)
gcmt = pd.read_csv(f_gcmt)
strike = []
dip = []
rake = []
kk = 0
dup = 0
for ii in range(len(neic.datenum)):
	dt = np.abs(gcmt.datenum.values - neic.datenum[ii])
	min_dt = np.min(dt)
	pos = np.argmin(dt)
	if min_dt <= 0.0005:
		strike.append(gcmt.strike1[pos])
		dip.append(gcmt.dip1[pos])
		rake.append(gcmt.rake1[pos])
		kk = kk + 1
	else:
		strike.append(-999)
		dip.append(-999)
		rake.append(-999)

neic['strike'] = strike
neic['dip'] = dip
neic['rake'] = rake
neic.to_csv(f_neic)
print('NEIC %f' % (kk / len(neic.datenum) * 100))
print('GCMT %f' % (kk / len(gcmt.datenum) * 100))

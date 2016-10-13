import pandas as pd
import numpy as np

location=[]
for i in range(1,7):
	for j in range(1,9):
		location.append((i,j))
heading=[(0,1),(0,-1),(-1,0),(1,0)]
Fs=[True, False]
Rs=[True,False]
Ls=[True,False]

state=[]
for i in location:
	for j in heading:
		for k in Fs:
			for l in Rs:
				for m in Ls:
					state.append((i,j,k,l,m))

rewardlist=pd.DataFrame(index=state, columns=['forward', 'right','left',None])

Qlist=rewardlist.where(rewardlist.notnull(), 0)


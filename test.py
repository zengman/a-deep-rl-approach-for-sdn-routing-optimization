import pandas as pd
import numpy as np
 
head = ["表头1" , "表头2" , "表头3"]
l = [[1 , 2 , 3],[4,5,6] , [8 , 7 , 9]]
l1 = [[0 , 1 , 0.0166667],[4,5,6] , [8 , 7 , 9]]
df = pd.DataFrame (l , columns = head)
newdf = pd.DataFrame (l1, columns = head)
df.to_csv ("testfoo.txt" , encoding = "utf-8")
with open('testfoo.txt', 'a') as f:
    newdf.to_csv(f, header=False)
# df2 = pd.read_csv ("testfoo.csv" , encoding = "utf-8")
# print (df2)
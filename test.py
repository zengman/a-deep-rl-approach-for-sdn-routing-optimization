import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd
tips = pd.read_csv('tips.csv')  
sns.set(style="ticks")                                     #设置主题  
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")   #palette 调色板  
plt.show()  
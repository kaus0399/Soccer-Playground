import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


'---------------------------------------------------------------------------------------------------------------------'


'''data = {'opponent': ['Chelsea','Cardiff','Huddersfield','Newcastle','Wolverhampton Wanderers'],
        'Home or Away': ['H', 'A', 'H', 'A', 'H',],
        'xG': [1.31, 2.42, 3.59, 1.48, 1.7],
        'shots': [15,17,21,11,13],
        'goals scored': [2, 2, 5, 3, 2,]}'''


data = pd.read_excel('/Users/kaustubh/Downloads/LiverpoolData.xlsx')
#print(data)
df =pd.DataFrame(data)



opponents = {'WestHam' :1 ,'CrystalPalace' :2,'Brighton' :3,'Leicester' :4,'Tottenham':5,'Southampton':6,'Chelsea':7,'ManchesterCity':8,'Huddersfield':9,'Cardiff':10,
'Arsenal':11,'Fulham':12,'Watford':13,'Everton':14,'Burnley':15,'Bournemouth':16, 'ManchesterUnited':17,'WolverhamptonWanderers':18,'NewcastleUnited':19, 'SheffieldUnited':20,
'Norwich':21, 'AstonVilla':22}   

df['HomeorAway'] = df['HomeorAway'].astype('category').cat.codes
data.opponent = [opponents[item] for item in data.opponent] 

#print(df)

plt.matshow(df.corr())
plt.xticks(np.arange(5), df.columns, rotation=90)
plt.yticks(np.arange(5), df.columns, rotation=0)
plt.colorbar()
#plt.show()

'---------------------------------------------------------------------------------------------------------------------'

X = np.asarray(df[['opponent', 'HomeorAway', 'xG', 'shots', 'shots on target', 'DEEP', 'PPDA', 'Corners', 'Half Time Goals', 'Posession']])
Y = np.asarray(df['goals scored'])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle= True, random_state=42)



lineReg = LinearRegression()
lineReg.fit(X_train, y_train)
print('Score: ', lineReg.score(X_test, y_test))
print('Weights: ', lineReg.coef_)

plt.plot(lineReg.predict(X_test))
plt.plot(y_test)
#plt.show()


'---------------------------------------------------------------------------------------------------------------------'


from sklearn import linear_model
reg = linear_model.Ridge (alpha = .2)
reg.fit(X_train, y_train)
print('Score: ', reg.score(X_test, y_test))
print('Weights: ', reg.coef_)

plt.plot(reg.predict(X_test))
plt.plot(y_test)
#plt.show()


'---------------------------------------------------------------------------------------------------------------------'


scores = []
coefs = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle= True, random_state=42)
    lineReg = LinearRegression()
    lineReg.fit(X_train, y_train)
    scores.append(lineReg.score(X_test, y_test))
    coefs.append(lineReg.coef_)
print('Linear Regression')
print(np.mean(scores))
print(np.mean(coefs, axis=0))
print('ok done linear')


scores = []
coefs = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle= True, random_state=42)
    lineReg = linear_model.Ridge (alpha = .5)
    lineReg.fit(X_train, y_train)
    scores.append(lineReg.score(X_test, y_test))
    coefs.append(lineReg.coef_)
print('\nRidge Regression')
print(np.mean(scores))
print(np.mean(coefs, axis=0))
print('ok done ridge \n')


'---------------------------------------------------------------------------------------------------------------------'

#home is 1 and 0 is away

print ("Linear Regression indicates Liverpool will score " + str((lineReg.predict([[4, 0, 3.77,	15,	6,	6,	11.43,	8,	1,	58]]))) + " goals vs Leicester - Away \n" )
print("Ridge Regression indicates Liverpool will score " + str((reg.predict([[4, 0, 3.77,	15,	6,	6,	11.43,	8,	1,	58 ]]))) + " goals vs Leicteser - Away \n")

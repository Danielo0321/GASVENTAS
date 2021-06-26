# GAS VENTAS

> Este es un proyecto empleado para la exploración de modelos y relaciones establecidas entre las 102 variables asociadas a un equipo catalogado como crítico dentro de una organización. El archivo suministrado contiene la evolución temporal de cada una de las variables, donde los registros fueron tomados cada 20 minutos para el periodo de tiempo comprendido entre el 01-Ene-2018 hasta el 10-May-2020.

> [**Pueden descargar el archivo .csv de este link**](https://drive.google.com/file/d/1LruXV-BIohTmBBgtYQqmETWbXoPj_edz/view?usp=sharing)


> Procedemos inicialmente a cargar las librerías que serán utilizadas

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sb


from statistics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import preprocessing
%matplotlib inline
```

> Seguidamente se realiza el cargue del dataframe (df) proveniente del archivo 'mi_df.csv'

```python
df = pd.read_csv('mi_df.csv', header=0)
```

> Luego una ligera inspección de los datos a ser analizados

```python
df.head()
```

> En total son 103 columnas, cada una representa una variable del equipo siendo que la primera columna corresponde a la estampa de tiempo (Todas las variables tienen la misma estampa de tiempo) y el volumen total de los datos es de 3GB apoximadamente.


```python
df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3719520 entries, 0 to 3719519
Columns: 103 entries, Unnamed: 0 to GASVENTAS105[, "Value"]
dtypes: float64(101), int64(1), object(1)
memory usage: 2.9+ GB
```

```python
df.describe()
```

> Poner bastante atención a las variables con mayor STD. Adicionalmente es posible elminar la variable GASVENTAS93[, "Value"] dado que no tiene asociado ningún valor y no le aporta nada al análisis.

```python
del(df['GASVENTAS93[, "Value"]'])
```

> Como no se puede visualizar el dataframe en su totalidad, procedemos a usar algunos comandos para verificar la existencia de datos **Null** o **NaN**

```python
df.columns[df.isnull().any()]
```
> El resultado indica las columnas con datos categorizados como  Null o NaN

```python
Index(['GASVENTAS30[, "Value"]'], dtype='object')
```
> Tambien es posible conocer la cantidad de datos que cumplen con la condición de búsqueda

```python
df.isnull().sum().sum()
```

```python
554513
```
> Indicando que existen al menos 554513 registros que deben ser corregidos o eliminados. Tambien es posible ir directamente al valor junto con su respectiva posición mediante el siguiente comando:

```python
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
print(df[df.isnull().any(axis=1)][null_columns].head())
```

```python
   GASVENTAS30[, "Value"]
0                     NaN
1                     NaN
2                     NaN
3                     NaN
4                     NaN
```

> Debido al elevado número de registros conflictivos no es recomendable eliminarlos dado que este procedimiento eliminaría la fila completa, no apenas de esta variables sino de las demás. En ese contexto, quedan abiertas las siguientes dos posibilidades:

- Eliminar toda la columna al igual que se hizo con la variable 'GASVENTAS393[, "Value"]', mediante el siguiente comando:
```python
del(df['GASVENTAS30[, "Value"]'])
```

- Atribuirle el valor de la media de la variable, a esos registros que presentan conflicto.
```python
df = df.fillna(df.mean())
```

> Se crea un nuevo dataframe, esta vez para suprimir la columna relacionada con la estampa de tiempo. Por el momento no será relevante esta columna dentro de los análisis.

```python
df1=df.iloc[:,2:]
# df1.head()
```

> Para mayor facilidad en la manipulación de las variables, se recomienda cambiar los nombres de cada una de ellas dado que los nombre originales pueden presentar conflictos en la ejecución de algunos comandos.

```python
df1 = df1.rename(columns={'Value':'GASVENTAS1', 'GASVENTAS2[, "Value"]':'GASVENTAS2', 'GASVENTAS3[, "Value"]':'GASVENTAS3', 'GASVENTAS4[, "Value"]':'GASVENTAS4', 'GASVENTAS5[, "Value"]':'GASVENTAS5', 'GASVENTAS6[, "Value"]':'GASVENTAS6', 'GASVENTAS7[, "Value"]':'GASVENTAS7', 'GASVENTAS8[, "Value"]':'GASVENTAS8', 'GASVENTAS9[, "Value"]':'GASVENTAS9', 'GASVENTAS10[, "Value"]':'GASVENTAS10', 'GASVENTAS11[, "Value"]':'GASVENTAS11', 'GASVENTAS12[, "Value"]':'GASVENTAS12', 'GASVENTAS13[, "Value"]':'GASVENTAS13', 'GASVENTAS14[, "Value"]':'GASVENTAS14', 'GASVENTAS15[, "Value"]':'GASVENTAS15', 'GASVENTAS16[, "Value"]':'GASVENTAS16', 'GASVENTAS17[, "Value"]':'GASVENTAS17', 'GASVENTAS18[, "Value"]':'GASVENTAS18', 'GASVENTAS19[, "Value"]':'GASVENTAS19', 'GASVENTAS20[, "Value"]':'GASVENTAS20', 'GASVENTAS21[, "Value"]':'GASVENTAS21', 'GASVENTAS22[, "Value"]':'GASVENTAS22', 'GASVENTAS23[, "Value"]':'GASVENTAS23', 'GASVENTAS24[, "Value"]':'GASVENTAS24', 'GASVENTAS25[, "Value"]':'GASVENTAS25', 'GASVENTAS26[, "Value"]':'GASVENTAS26', 'GASVENTAS27[, "Value"]':'GASVENTAS27', 'GASVENTAS28[, "Value"]':'GASVENTAS28', 'GASVENTAS29[, "Value"]':'GASVENTAS29', 'GASVENTAS30[, "Value"]':'GASVENTAS30', 'GASVENTAS31[, "Value"]':'GASVENTAS31', 'GASVENTAS32[, "Value"]':'GASVENTAS32', 'GASVENTAS33[, "Value"]':'GASVENTAS33', 'GASVENTAS34[, "Value"]':'GASVENTAS34', 'GASVENTAS35[, "Value"]':'GASVENTAS35', 'GASVENTAS36[, "Value"]':'GASVENTAS36', 'GASVENTAS37[, "Value"]':'GASVENTAS37', 'GASVENTAS38[, "Value"]':'GASVENTAS38', 'GASVENTAS39[, "Value"]':'GASVENTAS39', 'GASVENTAS40[, "Value"]':'GASVENTAS40', 'GASVENTAS41[, "Value"]':'GASVENTAS41', 'GASVENTAS42[, "Value"]':'GASVENTAS42', 'GASVENTAS43[, "Value"]':'GASVENTAS43', 'GASVENTAS44[, "Value"]':'GASVENTAS44', 'GASVENTAS45[, "Value"]':'GASVENTAS45', 'GASVENTAS46[, "Value"]':'GASVENTAS46', 'GASVENTAS47[, "Value"]':'GASVENTAS47', 'GASVENTAS48[, "Value"]':'GASVENTAS48', 'GASVENTAS49[, "Value"]':'GASVENTAS49', 'GASVENTAS50[, "Value"]':'GASVENTAS50', 'GASVENTAS51[, "Value"]':'GASVENTAS51', 'GASVENTAS52[, "Value"]':'GASVENTAS52', 'GASVENTAS53[, "Value"]':'GASVENTAS53', 'GASVENTAS54[, "Value"]':'GASVENTAS54', 'GASVENTAS55[, "Value"]':'GASVENTAS55', 'GASVENTAS56[, "Value"]':'GASVENTAS56', 'GASVENTAS57[, "Value"]':'GASVENTAS57', 'GASVENTAS58[, "Value"]':'GASVENTAS58', 'GASVENTAS59[, "Value"]':'GASVENTAS59', 'GASVENTAS60[, "Value"]':'GASVENTAS60', 'GASVENTAS61[, "Value"]':'GASVENTAS61', 'GASVENTAS62[, "Value"]':'GASVENTAS62', 'GASVENTAS63[, "Value"]':'GASVENTAS63', 'GASVENTAS64[, "Value"]':'GASVENTAS64', 'GASVENTAS65[, "Value"]':'GASVENTAS65', 'GASVENTAS66[, "Value"]':'GASVENTAS66', 'GASVENTAS67[, "Value"]':'GASVENTAS67', 'GASVENTAS68[, "Value"]':'GASVENTAS68', 'GASVENTAS69[, "Value"]':'GASVENTAS69', 'GASVENTAS70[, "Value"]':'GASVENTAS70', 'GASVENTAS71[, "Value"]':'GASVENTAS71', 'GASVENTAS72[, "Value"]':'GASVENTAS72', 'GASVENTAS73[, "Value"]':'GASVENTAS73', 'GASVENTAS74[, "Value"]':'GASVENTAS74', 'GASVENTAS75[, "Value"]':'GASVENTAS75', 'GASVENTAS76[, "Value"]':'GASVENTAS76', 'GASVENTAS77[, "Value"]':'GASVENTAS77', 'GASVENTAS78[, "Value"]':'GASVENTAS78', 'GASVENTAS79[, "Value"]':'GASVENTAS79', 'GASVENTAS80[, "Value"]':'GASVENTAS80', 'GASVENTAS81[, "Value"]':'GASVENTAS81', 'GASVENTAS82[, "Value"]':'GASVENTAS82', 'GASVENTAS83[, "Value"]':'GASVENTAS83', 'GASVENTAS84[, "Value"]':'GASVENTAS84', 'GASVENTAS85[, "Value"]':'GASVENTAS85', 'GASVENTAS86[, "Value"]':'GASVENTAS86', 'GASVENTAS87[, "Value"]':'GASVENTAS87', 'GASVENTAS88[, "Value"]':'GASVENTAS88', 'GASVENTAS89[, "Value"]':'GASVENTAS89', 'GASVENTAS90[, "Value"]':'GASVENTAS90', 'GASVENTAS91[, "Value"]':'GASVENTAS91', 'GASVENTAS92[, "Value"]':'GASVENTAS92', 'GASVENTAS95[, "Value"]':'GASVENTAS95', 'GASVENTAS99[, "Value"]':'GASVENTAS99', 'GASVENTAS100[, "Value"]':'GASVENTAS100', 'GASVENTAS101[, "Value"]':'GASVENTAS101', 'GASVENTAS102[, "Value"]':'GASVENTAS102', 'GASVENTAS103[, "Value"]':'GASVENTAS103', 'GASVENTAS104[, "Value"]':'GASVENTAS104', 'GASVENTAS105[, "Value"]':'GASVENTAS105',})
```

> Luego procedemos a analizar las dinámicas presentadas por algunas de las variables, observe con atención tanto la evolución temporal como los valores máximos y mínimos. 

```python
df1['GASVENTAS92'].plot()
```


![patients](Figuras_GV/GV92.png)


> Note la irregularidad del comportamiento presentado en la mitad del intervalo y que esa misma irregularidad está presente en las siguientes variables. Determinar con exactitud la fecha y consultar el la vitácora de los operadores si corresponde a alguna maniobra planeada.


```python
df1['GASVENTAS95'].plot()
```

![patients](Figuras_GV/GV95.png)


```python
df1['GASVENTAS101'].plot()
```

![patients](Figuras_GV/GV101.png)


```python
df1['GASVENTAS104'].plot()
```

![patients](Figuras_GV/GV104.png)


```python
df1['GASVENTAS13'].plot()
```

![patients](Figuras_GV/GV13.png)


> La variable anterior corresponde a la velocidad de la turbina. Su relevancia es bastante significativa dado que es un buen indicador de funcionamiento, en ese sentido puede considerarse que si la variable está por debajo de las 6000 rpm la turbina está en falla, si está en 0 rpm la turbina está fuera de línea, entre 8000 y 9000 la turbina está operando normalmente y por encima de 9000 rpm la turbina está en falla.



> Procederemos ahora a normalizar todas las variables.  

```python
# minmaxscaler, se usa como primera opcion
escalador=preprocessing.MinMaxScaler()
dse=escalador.fit_transform(df1)
df1=pd.DataFrame(dse, columns=df1.columns)
```

>  Una vez hecho esto, todas y cada una tendran escalas máximas y mínimas de 1 y 0 respectivamente, como se puede apreciar mediante el siguiente comando:

```python
df1.describe()
```




```python
df.columns
```

```python
df = df.rename(columns={'Value':'GASVENTAS1[, "Value"]'})
```





```python
df.head()
```

```python
%matplotlib inline
```
``` Python
df['Estado_rpm'] = ''
df.loc[df['GASVENTAS13[, "Value"]']==0, 'Estado_rpm'] = 1
df.loc[(df['GASVENTAS13[, "Value"]']>0) & (df['GASVENTAS13[, "Value"]']<6000), 'Estado_rpm'] = 2
df.loc[(df['GASVENTAS13[, "Value"]']>=6000) & (df['GASVENTAS13[, "Value"]']<8000), 'Estado_rpm'] = 3
df.loc[(df['GASVENTAS13[, "Value"]']>=8000) & (df['GASVENTAS13[, "Value"]']<9000), 'Estado_rpm'] = 4
df.loc[df['GASVENTAS13[, "Value"]']>=9000, 'Estado_rpm'] = 5
```

> Comprobamos que a todas las filas les fue atribuido alguno de los estados

```python
df['Estado_rpm'].value_counts()
``` 

```python
4    2543143
3     483779
5     470274
2     222209
1        115
Name: Estado_rpm, dtype: int64
``` 

> En un conjunto de datos de más de 100 variables, muy probablemente menos del 50% realiza un aporte significativo en la sintetización de modelos. Adicional a eso el costo computacional resultaría favorecido. Podemos implementar la desviación estándar para identificar aquellas variables cuyo comportamiento ha sido dinámico y por lo tanto su contribución en la extracción de modelos es mayor.

```Python
metricas = df.describe()
#metricas.info()
#std = metricas.iloc[2,:]
#print(std)
```

> Es posible graficar el vector correspondiente a la desviación estándar para visualizar las variables con comportamientos más dinámicos.

```Python
metricas.iloc[2,1:].plot(kind='bar', figsize=(20,10))
```

![patients](Figuras_GV/STD.png)

> Note que las variables con mayor desviación son GASVENTAS13[, "Value"], GASVENTAS14[, "Value"] y GASVENTAS101[, "Value"], siendo que GASVENTAS13[, "Value"], tal como se mencionó anteriormente corresponde a la velocidad de la turbina y por lo tanto es considerada una de las variables de mayor relevancia.

> Vamos ahora a crear los archivos de entrenamiento y validación, para eso he decidido tomar el 80% de los datos para entrenar y el 20% para validar


```Python
train, test = train_test_split(df, test_size=0.2)
```

> Note que efectivamente se han repartido los datos en las proporciones correspondientes

```Python
print('all:  ', len(df))
print('train:', len(train))
print('test: ', len(test))
```

```Python
all:   3719520
train: 2975616
test:  743904
```
> Finalmente se almacenan los archivos Gasv-train y Gasv-test

```Python
train_file = 'Gasv-train.csv'
pd.DataFrame.from_records(train).to_csv(train_file, index=False, header=True, sep=',')

test_file = 'Gasv-test.csv'
pd.DataFrame.from_records(test).to_csv(test_file, index=False, header=True, sep=',')
```

> Por tratarse de un ejercicio cuya salida corresponde a varios estados, no es posible usar los algoritmos de clasificación binaria

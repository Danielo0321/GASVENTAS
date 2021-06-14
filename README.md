# GAS VENTAS

> Este es un proyecto empleado para la exploración de modelos y relaciones establecidas entre las variables contenidas en el df llamado 'mi_df.csv'

> Pueden descargar los datos en el siguiente link:

**https://drive.google.com/file/d/1LruXV-BIohTmBBgtYQqmETWbXoPj_edz/view?usp=sharing**


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

```python
df = pd.read_csv('mi_df.csv', header=0)
```

```python
df.head()
```
> En total son 103 columnas, siendo que cada una representa una variable de la turbina y la primera columna corresponde a la estampa de tiempo cuyo intervalo comienza el 01/01/2018 a las 00h00.00 y termina el 10/05/2020 a las 23h59.59. Todas las variables tienen la misma estampa de tiempo y el periodo de muestreo es de 20seg.

```python
df.info()
```

> <class 'pandas.core.frame.DataFrame'>
RangeIndex: 3719520 entries, 0 to 3719519
Columns: 103 entries, Unnamed: 0 to GASVENTAS105[, "Value"]
dtypes: float64(101), int64(1), object(1)
memory usage: 2.9+ GB

```python
df.describe()
```

> Poner bastante atención a las variables con mayor STD y adicionalmente que la variable 93 no tiene asociado ningún valor (Descartarla dado que no aporta nada al análisis)

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

```python
df['GASVENTAS92[, "Value"]'].plot()
```

![patients](Figuras_GV/GV92.png)

```python
df['GASVENTAS95[, "Value"]'].plot()
```

![patients](Figuras_GV/GV95.png)

```python
df['GASVENTAS101[, "Value"]'].plot()
```

![patients](Figuras_GV/GV101.png)

```python
df['GASVENTAS104[, "Value"]'].plot()
```

![patients](Figuras_GV/GV104.png)

> La siguiente variable corresponde a la velocidad de la turbina. Su relevancia es bastante significativa dado que es un buen indicador del funcionamiento de la turbina y en ese sentido puede considerarse que si la variable está por debajo de las 6000 rpm la turbina está en falla, si está en 0 rpm la turbina está fuera de línea, entre 8000 y 8700 la turbina está operando normalmente y por encima de 8700 rpm la turbina está en falla.

```python
df['GASVENTAS13[, "Value"]'].plot()
```

![patients](Figuras_GV/GV13.png)

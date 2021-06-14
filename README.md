#GAS VENTAS

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

```python
df.info()
```

```python
df.describe()
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

> La siguiente variable corresponde a la velocidad de la turbina. Su relevancia es bastante significativa dado que es un buen indicador del funcionamiento de la turbina y en ese sentido puede considerarse que si la variable está por debajo de las 3000 rpm la turbina está en falla, si está en 0 rpm la turbina está fuera de línea, entre 8000 y 8700 la turbina está operando normalmente y por encima de ese rango la turbina está en falla.

```python
df['GASVENTAS13[, "Value"]'].plot()
```

![patients](Figuras_GV/GV13.png)

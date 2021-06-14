#GAS VENTAS

> Este es un proyecto empleado para la exploración de modelos y relaciones establecidas entre las variables contenidas en el df llamado 'mi_df.csv'

> Pueden descargar los datos en el siguiente link:

**https://drive.google.com/file/d/1LruXV-BIohTmBBgtYQqmETWbXoPj_edz/view?usp=sharing**


```P
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

```P
df = pd.read_csv('mi_df.csv', header=0)
```

```P
df.head()
```

```P
df.info()
```

```P
df.describe()
```

```P
df.columns
```

```P
df = df.rename(columns={'Value':'GASVENTAS1[, "Value"]'})
```

```P
df.head()
```

```P
%matplotlib inline
```

```P
df['GASVENTAS92[, "Value"]'].plot()
```

![patients](Figuras_GV/GV92.png)

```P
df['GASVENTAS95[, "Value"]'].plot()
```

![patients](Figuras_GV/GV95.png)

```P
df['GASVENTAS101[, "Value"]'].plot()
```

![patients](Figuras_GV/GV101.png)

```P
df['GASVENTAS104[, "Value"]'].plot()
```

![patients](Figuras_GV/GV104.png)

> La siguiente variable corresponde a la velocidad de la turbina. Su relevancia es bastante significativa dado que es un buen indicador del funcionamiento de la turbina y en ese sentido puede considerarse que si la variable está por debajo de las 3000 rpm la turbina está en falla, si está en 0 rpm la turbina está fuera de línea, entre 8000 y 8700 la turbina está operando normalmente y por encima de ese rango la turbina está en falla.

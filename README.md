#GAS VENTAS
Este es un proyecto empleado para la exploración de modelos y relaciones establecidas entre las variables contenidas en el df llamado 'mi_df.csv'

Pueden descargar los datos en el siguiente link:

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

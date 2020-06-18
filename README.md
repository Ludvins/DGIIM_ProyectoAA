# Ajuste de modelos de clasificación en la base de datos  [Forest Covertype](https://archive.ics.uci.edu/ml/datasets/covertype)

El objetivo es obtener modelos que sean capaces de predecir el tipo de cubierta forestal de una determinada región a partir de sus datos cartográficos. La base de datos proporciona estas observaciones, extraídas de cuatro zonas silvestres situadas en el Bosque Nacional Roosevelt, al norte de Colorado, junto con la cubierta forestal correspondiente. En estas zonas la acción del hombre es mínima, por lo que el tipo de cubierta forestal existente es resultado de procesos naturales.

Estamos, por tanto, frente a un problema de **aprendizaje supervisado**, en concreto de **clasificación multiclase**, donde la variable dependiente e toma valores en el rango 1,...,7. Cada uno de los valores codifica un tipo de cubierta forestal distinta. 

Debido a desbalanceos presentes en los datos, la métrica utilizada es el [Coeficiente de correlación de Matthews](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).

Los pasos seguidos son:
1. Creación de un conjunto de entrenamiento con una distribución de clases más equilibrada. Conjuntos de validación y test con distribución muy similar a la original.
2. Definición de 4 clases de modelos: lineal, random forest, SVM y multilayer perceptron.
3. Selección de parámetros mediante K-fold cross-validation, para seleccionar el mejor modelo de cada clase.
4. Entrenamiento de los 4 modelos resultantes en el conjunto de entrenamiento y selección del mejor resultado en validación.

El mejor modelo seleccionado ha sido: `RandomForest(class_weight='balanced', n_estimators=500, random_state=seed)`, donde el resultado obtenido en test es 0.71

| class |  precision  |  recall | f1-score  |  support |
|-------|-------------|---------|-----------|----------|
| 1     |   0.84      |  0.77   |   0.81    |  202999  |
| 2     |   0.85      | 0.84    |  0.84     |  271610  |
| 3     |   0.75      |  0.94   |   0.83    |  31766   |
| 4     |   0.81      |  0.81   |  0.81     |  2246    |
| 5     |   0.53      |  0.79   |  0.63     |  8240    |
| 6     |   0.67      |  0.74   |   0.70    |  15429   |
| 7     |   0.69      |  0.94   |  0.80     |  18222   |
|  accuracy   |          |          |    0.82  |  550512|
|    macro avg  |     0.73  |    0.83  |    0.78  |  550512|
| weighted avg   |   0.83   |  0.82  |    0.82  |  550512|

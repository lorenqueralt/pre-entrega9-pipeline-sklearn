# Pre-entrega · Pipeline de Preprocesamiento Robusto (Scikit-Learn)

**Curso:** Data Science II — Coderhouse

Encapsula todo el preprocesamiento y el modelo en un único objeto `Pipeline` de
Scikit-Learn, **evitando el data leakage** y asegurando que el mismo tratamiento se
aplique de forma consistente en entrenamiento e inferencia.

## Dataset

**Palmer Penguins** — 344 observaciones de pingüinos de 3 especies (Adelie, Chinstrap,
Gentoo) del archipiélago Palmer (Antártida). Se distribuye con el paquete
`palmerpenguins`.

| Tipo | Columnas |
|------|----------|
| Numéricas | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| Categóricas | `island`, `sex` |
| Target | `species` (clasificación multiclase) |

El dataset tiene **valores nulos reales** (2 en las medidas corporales, 11 en `sex`), lo
que permite demostrar la imputación dentro del pipeline.

## Estructura

```
preentrega9-pipeline-sklearn/
├── pipeline_penguins.ipynb   # notebook con todo el flujo (ejecutado)
├── modelo_pipeline.joblib    # pipeline entrenado y persistido
├── requirements.txt
├── .gitignore
└── README.md
```

## Cómo ejecutar

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook pipeline_penguins.ipynb
```

## Flujo del pipeline

1. **Carga y exploración** — identificación de columnas numéricas y categóricas.
2. **Train/Test split *antes* de cualquier ajuste** — los parámetros de preprocesamiento
   se aprenden solo con el train; el test queda intacto (evita **data leakage**).
3. **`ColumnTransformer`** con tratamiento diferenciado:
   - Numéricas → `SimpleImputer(median)` + `StandardScaler`.
   - Categóricas → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`.
4. **`Pipeline` final** = preprocesador + `LogisticRegression`.
5. **Evaluación** — accuracy y `classification_report` sobre el test.
6. **Inferencia** — `.predict()` sobre datos nuevos crudos, incluyendo un nulo y una
   categoría desconocida, para probar la robustez del pipeline.
7. **Persistencia** — exportación con `joblib` y verificación recargando el modelo.

## Cómo se evita el data leakage

- El `train_test_split` se ejecuta **antes** de imputar, escalar o codificar.
- Las transformaciones se ajustan (`fit`) únicamente dentro de `pipeline.fit(X_train, ...)`,
  por lo que medias, escalas y categorías provienen solo del entrenamiento.
- En inferencia, el pipeline aplica (`transform`) esos mismos parámetros, garantizando
  consistencia train/inferencia.

## Resultado

El modelo alcanza **accuracy = 1.00** en el test. Es un valor esperable: Palmer Penguins
es un dataset pequeño y las especies son casi linealmente separables por sus medidas, así
que una regresión logística las distingue sin dificultad. **No se trata de data leakage**:
el test nunca participó del ajuste del preprocesamiento ni del modelo.

## Librerías

| Librería | Uso |
|----------|-----|
| scikit-learn | Pipeline, ColumnTransformer, imputación, escalado, OneHot, modelo |
| palmerpenguins | dataset |
| joblib | persistencia del pipeline entrenado |
| pandas | manejo de datos |

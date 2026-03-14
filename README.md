# Proyecto: Conteo de Frutas Sanas vs Podridas con YOLO11

Este proyecto quedo unificado en una sola libreta y un solo script.

Objetivo:
- Descargar y preparar el dataset binario Healthy vs Rotten.
- Entrenar un clasificador YOLO11-cls sobre ese dataset.
- Ejecutar inferencia con detector + clasificador.
- Obtener conteos finales por imagen y por lote.

---

## Archivos principales

```text
Frutas_yolo_det.ipynb      <- Flujo completo en una sola libreta
frutas_pipeline.py         <- Script para ejecutar desde terminal
fruit_binary_yolo_cls/     <- Dataset binario generado
runs/classify/             <- Checkpoints de entrenamiento
yolo11n-cls.pt             <- Pesos base del clasificador
README.md                  <- Documentacion del proyecto
```

Las tres libretas antiguas fueron eliminadas para dejar un flujo unico y mas simple de mantener.

---

## Que hace la libreta unica

`Frutas_yolo_det.ipynb` incluye todo el pipeline:

1. Instalacion de dependencias.
2. Configuracion general del proyecto.
3. Descarga del dataset desde Kaggle.
4. Conversion del dataset a binario (`Healthy` y `Rotten`).
5. Split en `train`, `val` y `test`.
6. Entrenamiento simple de `yolo11n-cls.pt`.
7. Evaluacion en validacion y test.
8. Prueba automatica con una imagen del split `test`.
9. Visualizacion de metricas con `matplotlib`.
10. Inferencia batch sobre una carpeta completa.

---

## Dataset

Dataset usado:
- `muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten`

Estructura original:
- `Apple_Healthy`, `Apple_Rotten`, `Banana_Healthy`, `Banana_Rotten`, etc.

Estructura generada:

```text
fruit_binary_yolo_cls/
   train/
      Healthy/
      Rotten/
   val/
      Healthy/
      Rotten/
   test/
      Healthy/
      Rotten/
```

---

## Entrenamiento

Configuracion actual:
- Modelo base: `yolo11n-cls.pt`
- Nombre del run: `fruit_hs_vs_rt_cls_simple_ft5`
- Tamano de imagen: `640`
- Epocas: `5`
- Batch: `32`
- Augmentacion: se usa la augmentacion por defecto de YOLO para clasificacion

Checkpoint esperado:

```text
runs/classify/fruit_hs_vs_rt_cls_simple_ft5/weights/best.pt
```

Artefactos de metricas generados por el script:

```text
runs/classify/fruit_hs_vs_rt_cls_simple_ft5/metrics_summary.csv
runs/classify/fruit_hs_vs_rt_cls_simple_ft5/metrics_summary.png
```

---

## Inferencia

La inferencia usa un enfoque hibrido:

1. `yolo11n.pt` detecta objetos en la imagen.
2. Cada bounding box se recorta.
3. El clasificador entrenado decide si el recorte es `Healthy` o `Rotten`.
4. Se genera el conteo final por imagen.

Clases COCO filtradas por defecto:
- `apple`
- `banana`
- `orange`
- `broccoli`
- `carrot`

Salida esperada:
- Imagen anotada con cajas y etiquetas.
- Conteo final de sanas y podridas.
- CSV con conteos en modo batch.

---

## Uso de la libreta

Abrir y ejecutar en orden:
- `Frutas_yolo_det.ipynb`

Resultado esperado:
- Dataset binario generado.
- Checkpoint `best.pt` entrenado.
- Tabla de metricas en `val` y `test`.
- Grafica de metricas con `matplotlib`.
- Prueba con una imagen tomada automaticamente desde `DATA_DIR/test`.
- Inferencia batch con `counts.csv`.

---

## Uso del script

Ejemplos:

Preparar dataset:

```powershell
python frutas_pipeline.py --prepare-dataset
```

Entrenar clasificador:

```powershell
python frutas_pipeline.py --train
```

Preparar dataset y entrenar:

```powershell
python frutas_pipeline.py --all
```

Inferencia sobre una imagen:

```powershell
python frutas_pipeline.py --image sample.jpg --output-image salida.jpg
```

Prueba automatica con una imagen del split test:

```powershell
python frutas_pipeline.py --demo-image --output-image sample_result.jpg
```

Inferencia batch sobre una carpeta:

```powershell
python frutas_pipeline.py --input-dir inference_images --output-dir inference_outputs
```

Opciones utiles:
- `--run-name` para cambiar el nombre del entrenamiento.
- `--epochs` para cambiar las epocas.
- `--batch` para cambiar el batch size.
- `--demo-image` para probar rapido el modelo con una imagen del split test.

---

## Requisitos

- Python 3.10+
- Entorno virtual recomendado: `.venv`

Dependencias principales:
- `kagglehub`
- `tqdm`
- `ultralytics`
- `opencv-python`
- `matplotlib`
- `pandas`
- `scikit-learn`

La libreta ya incluye la instalacion de dependencias. El script asume que ya estan instaladas.

---

## GPU

Comportamiento actual:
- Si hay GPU CUDA disponible, se usa automaticamente.
- Si no hay GPU compatible, se usa CPU.

Nota:
- En Windows con GPU AMD, Ultralytics normalmente termina usando CPU en este flujo.
- Si quieres AMD con aceleracion real, la opcion recomendable es Linux o WSL2 con ROCm.

---

## Troubleshooting

- Si falla Kaggle, revisa tu token de API.
- Si no aparece `best.pt`, revisa que el entrenamiento termino correctamente.
- Si la inferencia no encuentra el clasificador, revisa la ruta del run.
- Si va lento, probablemente estas corriendo en CPU.
- Si el detector pierde objetos, baja `det_conf` o desactiva el filtro de clases.

---

## Creditos

- Ultralytics YOLO11
- Kaggle dataset: `muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten`

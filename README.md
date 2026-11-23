# Práctica 2 · Autoencoders (AA2)

Eliminación de ruido y superresolución sobre **MNIST** con autoencoders, partiendo del código de clase (`codigo_clase/`).

**Entrega:** domingo 6 de diciembre de 2026.

## Estructura

| Notebook | Contenido | Nivel | Responsable |
|---|---|---|---|
| `01_ruido_basico.ipynb` | Ruido gaussiano (σ = 0.5). Se entrenan el autoencoder **denso** y el **convolucional** de clase, se comparan por MSE en test y se elige el mejor | Básico | Compañero (CPU) |
| `02_ruido_niveles.ipynb` | El convolucional entrenado con σ = 0.1, 0.3, 0.5, 0.7 y 1.0: tabla y gráfica de MSE según σ | Avanzado | Daniel (GPU) |
| `03_superres_basico.ipynb` | Superresolución 14×14 → 28×28 y 7×7 → 28×28 (convolucional y denso) | Básico | Compañero (CPU) |
| `04_superres_otras.ipynb` | Otras resoluciones de entrada (4, 10, 21 → 28) y de salida (7 → 14, 14 → 56) | Avanzado | Daniel (GPU) |

Cada notebook es independiente: lleva sus imports, la carga de MNIST, el modelo, el entrenamiento y la figura de resultados.
Las ejecuciones con más entrenamientos (02 y 04) van en la GPU (RTX 4050). Las del nivel básico (01 y 03) son pocas y caben en CPU.

## Criterios comunes (para que todo sea coherente)

- **Base:** el código de clase. Se mantienen `MSELoss`, `Adam(lr=0.001)`, `batch_size=64`, `epochs=10` y `torch.manual_seed(42)`.
- **Figura obligatoria del enunciado:** matriz de **3 filas × 10 columnas**: 1) original, 2) con ruido o reducida, 3) salida del autoencoder. Se usa la función `mostrar_resultados(...)` de `02_ruido_niveles.ipynb` (cópiala tal cual).
- **Ruido:** `torch.clamp(images + sigma * torch.randn_like(images), 0, 1)`. Se añade lote a lote en el bucle de entrenamiento. Entrada: imagen con ruido; objetivo: imagen original.
- **Reducir resolución:** `v2.functional.resize(images, [n, n], antialias=True)` aplicado al lote dentro del bucle. Entrada: imagen pequeña; objetivo: imagen original de 28×28.
- **Arquitectura de superresolución (convolucional):**
  - Encoder: `Conv2d(1→16, k=3, s=1, p=1)` + ReLU, `Conv2d(16→32, k=3, s=1, p=1)` + ReLU (mantienen el tamaño n×n porque la entrada ya es pequeña), `Flatten` y `Linear(32·n·n → 64)`.
  - Decoder: **el mismo de clase** (`Linear(64 → 32·7·7)`, `Unflatten`, `ConvTranspose2d` 7→14 y 14→28, `Sigmoid`).
- **Métrica:** MSE medio en el conjunto de test frente a la imagen original.
- **Comentarios:** en español y breves, al estilo del código de clase. El enunciado no los prohíbe.

## Reparto y calendario

### Daniel (GPU)
- [x] **Lun 23 nov:** montar el repositorio, el README y `02_ruido_niveles.ipynb` ejecutado.
- [ ] **Sáb 28 nov:** `04_superres_otras.ipynb` ejecutado.
- [ ] **Lun 30 nov (prácticas AA2):** revisar juntos el `01` y unificar criterios.
- [ ] **Mié 2 dic:** revisar el `03`, escribir las conclusiones de 02 y 04 y completar este README.
- [ ] **Vie 4 dic:** reiniciar y ejecutar todos los notebooks de principio a fin, y preparar el guion de defensa.
- [ ] **Sáb 5 dic:** entregar (un día antes del plazo).

### Compañero (CPU)
- [ ] **Antes del lun 30 nov:** `01_ruido_basico.ipynb` ejecutado, con la conclusión de que el convolucional es mejor que el denso.
- [ ] **Antes del mié 2 dic:** `03_superres_basico.ipynb` ejecutado (14→28 y 7→28).
- [ ] **Vie 4 dic:** leer y entender el 02 y el 04 (en la defensa os pueden preguntar por cualquier parte).

## Cómo trabajamos con Git

Cada uno toca **solo sus notebooks**, así que no hay conflictos y se puede trabajar directamente en `main`:

```bash
git pull                      # antes de empezar
git add 01_ruido_basico.ipynb
git commit -m "01: ruido básico denso vs convolucional"
git pull && git push
```

Sube los notebooks **ya ejecutados**, con las salidas y las figuras visibles. La carpeta `data/` (MNIST) no se sube.

## Requisitos

`torch`, `torchvision` y `matplotlib`, las mismas librerías que en clase.

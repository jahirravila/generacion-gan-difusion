# Generación — DCGAN y fine-tuning de Stable Diffusion con diffusers

**Repositorio:** URL_DEL_REPOSITORIO

| # | Notebook | Contenido | Estado |
|---|---|---|---|
| 01 | `01_DCGAN_rostros.ipynb` | DCGAN en PyTorch que genera rostros 64×64 (base ORL): entrenamiento adversarial, evolución, interpolación latente, memorización y diversidad | Ejecutado (CPU, 10 min) |
| 02 | `02_StableDiffusion_mini_LoRA.ipynb` | Stable Diffusion en miniatura con las clases de diffusers (AutoencoderKL, CLIPTextModel, UNet2DConditionModel, DDPM/DDIM, StableDiffusionPipeline) + fine-tuning LoRA (PEFT) del concepto «rostro» | Ejecutado (CPU, 25 min) |
| 03 | `03_StableDiffusion15_LoRA_Colab.ipynb` | Fine-tuning LoRA de Stable Diffusion 1.5 con el dataset `lambdalabs/naruto-blip-captions` | Para Colab con GPU T4 |

## Resultados principales
**DCGAN (400 épocas):** rostros nuevos reconocibles; sin copias del dataset (distancia al vecino real 17.8 frente a 12.4 entre reales) y sin mode collapse (diversidad 27.1 frente a 27.3).

**Mini Stable Diffusion + LoRA (44 032 parámetros entrenables, 181 KB):**

| Métrica | Antes de LoRA | Después de LoRA |
|---|---|---|
| «foto de un rostro» reconocida como rostro | 0 % | 93.8 % |
| Adherencia al prompt en las 10 prendas | 87.5 % | 85.0 % |

## Ejecución
* Notebooks 01 y 02: Colab (`Entorno de ejecución → Ejecutar todas`), CPU o GPU. Los datos se descargan solos.
* Notebook 03: requiere GPU (`Cambiar tipo de entorno → T4`) y acceso a Hugging Face Hub; ≈ 30 min.

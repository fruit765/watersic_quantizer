# Инструкция по квантованию Huihui-Ornith-1.5-35B-A3B в WaterSIC на Kaggle (2x T4 GPU)

Данное руководство описывает полный процесс квантования мультимодальной MoE-модели **`huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated`** (~35.8B параметров, 70 ГБ в BF16) в оптимизированный контейнер **`.wsic`** в бесплатной среде **Kaggle Notebooks** с 2 графическими процессорами NVIDIA T4 (2x 16 ГБ VRAM) и 30 ГБ оперативной памяти (CPU RAM).

---

## 1. Архитектура памяти и параллелизм на 2x T4 GPU

- **Проблема**: Полная модель в FP16/BF16 весит **~70 ГБ**. Загрузить её целиком в RAM (лимит 30 ГБ) или VRAM (лимит 16 ГБ на одну T4) невозможно — произойдет мгновенный `OOM Crash`.
- **Решение**: Прогрессивная послойная схема (**Streaming Layer-by-Layer Execution**):
  1. Модель инициализируется на CPU через `low_cpu_mem_usage=True` и `accelerate`. Веса подгружаются лениво (lazy mmap).
  2. Входные калибровочные токенты прогоняются через `embed_tokens`, а скрытые состояния $X$ сохраняются на системный диск / RAM (~1.5 ГБ для 64 пассажей по 2048 токенов).
  3. В память GPU поочередно загружается **только один текущий слой** MoE (~1.8–2.2 ГБ).
  4. Собираются матрицы ковариаций $H = \frac{1}{N} X^T X$ для всех линейных модулей слоя (проекции внимания, shared expert и 64 routed experts).
  5. **Параллелизм 2x GPU**: Квантование 64 экспертов слоя распараллеливается между двумя T4:
     - GPU 0 (`cuda:0`): четные эксперты (0, 2, 4...) + внимание;
     - GPU 1 (`cuda:1`): нечетные эксперты (1, 3, 5...).
     - Это снижает время факторизации в 2 раза!
  6. Слой квантуется алгоритмом **WaterSIC LDLQ**, веса сбрасываются в `.wsic`, а старый слой немедленно удаляется из VRAM через `gc.collect()` и `torch.cuda.empty_cache()`.
  7. **Пиковое потребление VRAM**: **~6–9 ГБ** на каждой T4.
  8. **Пиковое потребление CPU RAM**: **~12–16 ГБ** (из доступных 30 ГБ).

---

## 2. Учет формата диалога (Chat Template) при калибровке

Для модели `huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated` критически важно калибровать веса на **нативных диалоговых токенах**:
- Применяется `tokenizer.apply_chat_template` с системным промптом, тегами `<|im_start|>` / `<|im_end|>` и блоками рассуждений `<think>`.
- Это обеспечивает точную активацию экспертов логики (reasoning experts) и роутера, исключая деградацию модели при реальном общении.

---

## 2. Настройка Kaggle Notebook

1. Создайте новый Notebook на [Kaggle](https://www.kaggle.com/).
2. В панели справа (**Settings**):
   - **Accelerator**: Выберите **`GPU T4 x2`**.
   - **Internet**: Переключите в положение **`Always on`** (необходимо для скачивания весов модели с Hugging Face).
   - **Persistence**: Переключите в **`Variables and Files`**.
3. Добавьте секрет с Hugging Face токеном (если модель или датасеты требуют авторизации):
   - В меню: `Add-ons` -> `Secrets` -> добавьте ключ `HF_TOKEN`.

---

## 3. Выполнение в ячейках Kaggle Notebook

### Ячейка 1: Установка зависимостей
```bash
%%bash
pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate safetensors datasets xxhash huggingface_hub
```

### Ячейка 2: Загрузка файлов квантизатора
Загрузите папку `watersic_quantizer/` в корень ноутбука (или клонируйте ваш репозиторий).
> **Примечание**: Все алгоритмы WaterSIC (декомпозиция Cholesky, water-filling и упаковка кодов) уже встроены в `watersic_quantizer/core/zsic.py`, клонировать внешние репозитории WaterSIC больше не требуется!

### Ячейка 3: Запуск процесса квантования
```python
import os
import torch

# Оптимизация фрагментации памяти PyTorch на T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from watersic_quantizer.quantize import run_quantization

model_id = "huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated"
profile_path = "watersic_quantizer/profiles/qwen35_moe_35b.json"
output_wsic = "/kaggle/working/Huihui-Ornith-1.5-35B-A3B-WaterSIC.wsic"

run_quantization(
    model_id=model_id,
    profile_path=profile_path,
    output_wsic=output_wsic,
    nsamples=64,   # 64 пассажа по 2048 токенов = 131 072 калибровочных токена
    seqlen=2048,
    device="cuda:0"
)
```

---

## 4. Распределение квантов в профиле `qwen35_moe_35b.json`

| Компонент модели | Имена тензоров | Целевая битность | Обоснование |
| :--- | :--- | :---: | :--- |
| **Нормализации** | `attn_norm`, `ffn_norm`, `output_norm` | **FP16 / BF16 (0b)** | Критичны для численной стабильности слоя |
| **Роутеры экспертов**| `ffn_gate_inp` / `gate` | **FP16 / BF16 (0b)** | Предотвращение коллапса маршрутизации токенов |
| **Biases & RoPE/SSM**| `bias`, `rotary_emb`, `ssm_*` | **FP16 / BF16 (0b)** | 1D-параметры, не дают выигрыша по объему |
| **Эмбеддинги токенов**| `token_embd.weight` / `embed_tokens` | **4 бита WaterSIC** | Высокая точность входных представлений |
| **Выходная голова (LM Head)**| `output.weight` / `lm_head` | **6 бит WaterSIC** | Точное ранжирование логитов словаря |
| **Проекции внимания**| `q_proj`, `k_proj`, `v_proj`, `o_proj` | **4 бита WaterSIC** | Сохранение структуры внимания |
| **Shared Experts** | `shared_expert.*` | **4 бита WaterSIC** | Активны на каждом токене, требуют высокой точности |
| **Routed Experts (64 шт)**| `experts.*` (`gate_proj`, `up_proj`, `down_proj`) | **3 бита WaterSIC** | Занимают 85% весов модели, идеальны для 3 бит |
| **Фотоэнкодер (Vision)**| `visual.*` / `model.visual.*` | **6 бит WaterSIC** | Сохранение 100% точности OCR и мелких деталей |

---

## 5. Скачивание готового файла `.wsic`

После завершения файл модели размером **~15–18 ГБ** (вместо исходных 70 ГБ) будет сохранен в `/kaggle/working/`:
1. В правом меню Kaggle перейдите на вкладку **Output**.
2. Нажмите **`Save Version`** -> выберите **`Save & Run All (Commit)`**.
3. Либо нажмите на три точки рядом с `.wsic` файлом и выберите **`Download`**.
4. Для удобного скачивания больших файлов через CLI можно автоматически создать Kaggle Dataset:
   ```bash
   kaggle datasets create -p /kaggle/working/ -r tar
   ```

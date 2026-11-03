# Пошаговое руководство: Квантование Huihui-Ornith-1.5-35B-A3B в WaterSIC на Kaggle (2x NVIDIA T4)

Данное исчерпывающее руководство описывает полный процесс квантования мультимодальной MoE-модели **`huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated`** (~35.8 млрд параметров, исходный вес **~70 ГБ** в BF16) в ультракомпактный квантованный контейнер **`.wsic`** (~15–18 ГБ) в бесплатной облачной среде **Kaggle Notebooks** на двух GPU **NVIDIA Tesla T4 (2x 16 ГБ VRAM)** и 30 ГБ оперативной памяти (CPU RAM).

---

## 1. Архитектура и принцип работы на 2x T4

| Ресурс | Доступно на Kaggle | Использование квантизатором | Запас |
| :--- | :---: | :---: | :---: |
| **GPU 0 (`cuda:0`)** | 16 ГБ GDDR6 | **~5.5–8.0 ГБ** (Forward hooks + LDLQ Worker 0) | **~50% свободно** |
| **GPU 1 (`cuda:1`)** | 16 ГБ GDDR6 | **~4.0–6.5 ГБ** (LDLQ Worker 1) | **~60% свободно** |
| **CPU RAM** | 30 ГБ | **~12–16 ГБ** (активации калибровки + lazy mmap) | **~50% свободно** |
| **Диск `/kaggle/working`** | 20 ГБ | **~16–18 ГБ** (выходной `.wsic` файл) | **Хватает** |
| **Системный кэш `~/.cache`** | ~73 ГБ | **~70 ГБ** (веса исходной модели) | **Хватает** |

### Как распараллеливаются две видеокарты:
1. **Сбор ковариаций $H = \frac{1}{N} X^T X$ (Forward Hooks)**:
   - В память загружается **только один текущий слой** MoE (~1.8–2.2 ГБ).
   - Скрытые состояния $X$ прогоняются через слой на первичной карте `cuda:0`.
2. **Параллельная факторизация экспертов (WaterSIC LDLQ)**:
   - В каждом слое MoE находятся **64 маршрутизируемых эксперта**, у каждого по 3 матрицы (`gate_proj`, `up_proj`, `down_proj`) — суммарно **192 огромные матрицы на слой**!
   - Квантизатор запускает `ThreadPoolExecutor(max_workers=2)`:
     - **GPU 0 (`cuda:0`)**: параллельно квантует четные матрицы (`cuda:0` Worker);
     - **GPU 1 (`cuda:1`)**: параллельно квантует нечетные матрицы (`cuda:1` Worker).
   - Это сокращает время факторизации ровно в **2 раза** и загружает оба чипа T4 на 100%.
3. **Очистка памяти**:
   - После завершения слоя сжатые веса записываются в `.wsic`, а старый слой удаляется из памяти через `torch.cuda.empty_cache()` на обеих видеокартах.

---

## 2. Подготовка Kaggle Notebook

### Шаг 2.1: Создание блокнота
1. Перейдите на [Kaggle](https://www.kaggle.com/) и нажмите **`+ Create`** -> **`New Notebook`**.

### Шаг 2.2: Включение 2x T4 GPU и интернета
В правой панели настроек блокнота (**Notebook settings**):
1. **Accelerator**: выберите **`GPU T4 x2`** *(Обязательно! По умолчанию стоит None или P100)*.
2. **Internet**: переключите тумблер в положение **`Always on`** *(Без этого Kaggle не сможет скачивать веса модели с Hugging Face)*.
3. **Persistence**: переключите на **`Variables and Files`** *(Сохраняет рабочее состояние)*.

### Шаг 2.3: Настройка Hugging Face токена (Рекомендуется)
Если модель или датасеты требуют авторизации:
1. В верхнем меню блокнота: **`Add-ons`** -> **`Secrets`**.
2. Добавьте секрет с именем **`HF_TOKEN`** и вставьте ваш токен доступа с [HuggingFace Settings](https://huggingface.co/settings/tokens).

---

## 3. Пошаговые ячейки блокнота (Копировать и запускать)

### Ячейка 1: Проверка оборудования и оптимизация CUDA
Убедимся, что Kaggle выделил именно 2x T4, и включим режим `expandable_segments` для предотвращения фрагментации VRAM:

```python
import os
import torch

# Оптимизация фрагментации памяти PyTorch для предотвращения OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("CUDA доступна:", torch.cuda.is_available())
print("Количество GPU:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GB")

assert torch.cuda.device_count() >= 2, "ОШИБКА: Выберите Accelerator -> GPU T4 x2 в настройках справа!"
```

---

### Ячейка 2: Установка необходимых библиотек
Устанавливаем современные совместимые версии `transformers`, `accelerate` и движка калибровки:

```bash
%%bash
pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate safetensors datasets xxhash huggingface_hub
```

---

### Ячейка 3: Клонирование репозитория квантизатора
Клонируем автономный репозиторий `watersic_quantizer`:

```bash
%%bash
cd /kaggle/working
rm -rf watersic_quantizer
git clone https://github.com/fruit765/watersic_quantizer.git
cd watersic_quantizer
ls -la
```

---

### Ячейка 4: Проверка профиля квантования
Убедимся, что профиль `qwen35_moe_35b.json` настроен в точности под ваши требования:

```python
import json

profile_path = "/kaggle/working/watersic_quantizer/profiles/qwen35_moe_35b.json"
with open(profile_path, "r", encoding="utf-8") as f:
    prof = json.load(f)

print(f"Профиль: {prof['name']}")
print(f"Описание: {prof['description']}\n")
print(f"{'Компонент':<35} | {'Битность':<8} | Действие")
print("-" * 65)
for rule in prof["rules"]:
    print(f"{rule['description']:<35} | {str(rule['target_bits']) + ' bit':<8} | {rule['action']}")
```

---

### Ячейка 5: Запуск квантования на двух GPU параллельно
Запускаем послойный процесс квантования с одновременным использованием `cuda:0` и `cuda:1`:

```python
import sys
import os
import torch

# Принудительная перезагрузка модулей Python (гарантирует применение обновлений из git)
for mod in list(sys.modules.keys()):
    if "watersic_quantizer" in mod:
        del sys.modules[mod]

# Добавляем модуль квантизатора в пути поиска Python
sys.path.insert(0, "/kaggle/working/watersic_quantizer")

from watersic_quantizer.quantize import run_quantization

# Параметры квантования
MODEL_ID = "huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated"
PROFILE_PATH = "/kaggle/working/watersic_quantizer/profiles/qwen35_moe_35b.json"
OUTPUT_WSIC = "/kaggle/working/Huihui-Ornith-1.5-35B-A3B-WaterSIC.wsic"

# Запуск послойного квантования
run_quantization(
    model_id=MODEL_ID,
    profile_path=PROFILE_PATH,
    output_wsic=OUTPUT_WSIC,
    nsamples=64,                      # 64 пассажа по 2048 токенов = 131 072 калибровочных токена
    seqlen=2048,
    devices=["cuda:0", "cuda:1"]      # Использование двух T4 видеокарт параллельно!
)
```

> [!TIP]
> **Время выполнения**: Квантование 28 слоев MoE модели (64 эксперта на слой) на двух T4 занимает примерно **2.5–3.5 часа**. Лимит непрерывной работы ячейки Kaggle составляет **12 часов**, поэтому процесс гарантированно укладывается с большим запасом!

---

### Ячейка 6: Валидация готового контейнера `.wsic`
После окончания работы квантизатора проверяем целостность полученного бинарного контейнера:

```python
import os
import struct

output_wsic = "/kaggle/working/Huihui-Ornith-1.5-35B-A3B-WaterSIC.wsic"

if os.path.exists(output_wsic):
    size_gb = os.path.getsize(output_wsic) / (1024**3)
    print(f" УСПЕХ! Контейнер успешно создан.")
    print(f" Путь к файлу: {output_wsic}")
    print(f" Итоговый размер: {size_gb:.2f} ГБ (Сжатие с 70 ГБ в {70.0 / size_gb:.1f} раз!)")
    
    # Проверка сигнатуры заголовка WSIC
    with open(output_wsic, "rb") as f:
        magic = f.read(4)
        ver_major, ver_minor, n_layers, n_tensors, base_bits = struct.unpack("<HHIII", f.read(16))
        
    print(f"\nЗаголовок контейнера:")
    print(f" - Сигнатура Magic: {magic.decode('ascii', errors='ignore')} (Ожидается: WSIC)")
    print(f" - Версия формата:  v{ver_major}.{ver_minor}")
    print(f" - Количество слоев: {n_layers}")
    print(f" - Всего тензоров:  {n_tensors}")
    print(f" - Базовый битрейт: {base_bits} bit")
else:
    print(" Ошибка: Файл контейнера не найден!")
```

---

## 4. Как забрать готовый файл из Kaggle (16–18 ГБ)

Поскольку файл весит около 16–18 ГБ, скачивание через веб-браузер может прерваться из-за сетевого тайм-аута. Рекомендуется использовать один из двух надёжных способов:

### Вариант А: Прямая загрузка в Hugging Face (Самый быстрый и удобный)
На серверах Kaggle скорость исходящего интернет-канала составляет ~50–100 МБ/с. Загрузка 16 ГБ на Hugging Face займет всего 3–5 минут:

```python
from huggingface_hub import HfApi, create_repo
import os

# Получаем токен из секретов Kaggle или укажите строкой
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_your_token_here")
REPO_ID = "your-username/Huihui-Ornith-1.5-35B-A3B-WaterSIC"

api = HfApi(token=HF_TOKEN)

# 1. Создаем репозиторий модели (если еще не создан)
create_repo(repo_id=REPO_ID, token=HF_TOKEN, repo_type="model", exist_ok=True)

# 2. Быстрая загрузка .wsic контейнера
print(f"Загрузка файла в {REPO_ID}...")
api.upload_file(
    path_or_fileobj="/kaggle/working/Huihui-Ornith-1.5-35B-A3B-WaterSIC.wsic",
    path_in_repo="Huihui-Ornith-1.5-35B-A3B.wsic",
    repo_id=REPO_ID,
    token=HF_TOKEN
)
print("Загрузка на Hugging Face завершена!")
```

### Вариант Б: Сохранение в Kaggle Dataset
Если вы хотите оставить модель внутри инфраструктуры Kaggle:
```bash
%%bash
cd /kaggle/working
cat << 'EOF' > dataset-metadata.json
{
  "title": "Huihui Ornith 35B WaterSIC Quantized",
  "id": "yourusername/huihui-ornith-35b-watersic",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
kaggle datasets create -p /kaggle/working/
```

---

## 5. Сводка квантования профиля `qwen35_moe_35b`

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

## 6. Частые вопросы и устранение ошибок (FAQ / Troubleshooting)

### В: Что делать, если при старте появляется `OutOfMemoryError: CUDA out of memory`?
**О**: Это означает, что PyTorch не успел очистить кэш от предыдущих запусков. Выполните:
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```
Либо перезапустите сессию блокнота (**Restart Session** в правом верхнем углу).

### В: Хватит ли дискового пространства в `/kaggle/working`?
**О**: В Kaggle каталог `/kaggle/working` имеет квоту **~20 ГБ**. Итоговый `.wsic` файл для 35B модели занимает **~16.2 ГБ**, поэтому он полностью помещается на диск. Главное — не сохранять туда дубликаты весов в FP16.

### В: Как работает многодоменная калибровка?
**О**: Встроенный калибратор `calibrate.py` автоматически скачивает сбалансированную смесь токенов (WikiText-2, код Python, математические рассуждения GSM8k и русскоязычные статьи) и оборачивает их в нативный формат диалога модели (`tokenizer.apply_chat_template`), включая теги `<|im_start|>` и блоки рассуждений `<think>`. Это гарантирует равномерную активацию всех 64 экспертов и предотвращает их деградацию.

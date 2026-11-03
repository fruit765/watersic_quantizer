# WaterSIC Model Quantizer & WSIC Container Builder

Пакет для квантования больших языковых и мультимодальных моделей (Dense и MoE) в унифицированный контейнер **`.wsic` (WaterSIC Unified Model Container)** с поддержкой смешанного битрейта (Mixed-Bitrate: 3, 4, 6, 8 бит и FP16/BF16) и профилей квантования.

---

## Архитектура папки `watersic_quantizer`

```text
watersic_quantizer/
├── profiles/
│   ├── qwen35_moe_35b.json     # Профиль для huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated
│   └── qwen35_dense_08b.json   # Профиль для Qwen/Qwen3.5-0.8B
├── core/
│   ├── profile_engine.py       # Парсер профилей и сопоставление тензоров по regex
│   ├── packing.py              # Упаковка 3, 4, 6, 8-битных кодов и CSR-выбросов
│   └── wsic_writer.py          # Сборщик контейнера .wsic с выравниванием по 64 байта
├── kaggle/
│   ├── run_kaggle.sh           # Bash-скрипт запуска на Kaggle
│   └── kaggle_guide.md         # Руководство по квантованию на 2x T4 GPU
├── calibrate.py                # Генератор мультидоменного калибровочного датасета
├── quantize.py                 # Главный послойный прогрессивный квантизатор
├── verify_local.py             # Локальная верификация и тест битовых round-trip
└── README.md                   # Данный документ
```

---

## Быстрый старт

### 1. Локальная проверка целостности и правил профилей
```powershell
python watersic_quantizer/verify_local.py
```

### 2. Запуск квантования на локальной машине или сервере
```powershell
python watersic_quantizer/quantize.py `
    --model_id "Qwen/Qwen3.5-0.8B" `
    --profile "watersic_quantizer/profiles/qwen35_dense_08b.json" `
    --output_wsic "Qwen3.5-0.8B-Custom.wsic" `
    --nsamples 64 `
    --seqlen 1024
```

### 3. Запуск на Kaggle для 35B MoE модели
См. подробную пошаговую инструкцию в [`watersic_quantizer/kaggle/kaggle_guide.md`](file:///F:/projects/megastrata/watersic_quantizer/kaggle/kaggle_guide.md).

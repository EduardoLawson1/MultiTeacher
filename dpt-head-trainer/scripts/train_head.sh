#!/bin/bash
# Exemplo de como executar o head_training.py

cd /home/pdi5060ti/Documentos/MultiTeacher/dpt-head-trainer

# Treino supervisionado em todas as classes PASCAL VOC
python head_training.py \
    --config configs/pascal.yaml \
    --save-path training-logs/pascal_full \
    --pretrained None

# Ou para treinar apenas um grupo de classes:
# python head_training.py \
#     --config configs/pascal.yaml \
#     --save-path training-logs/pascal_vehicles \
#     --class-group vehicles \
#     --pretrained None

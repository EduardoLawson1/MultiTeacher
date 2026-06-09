# Head Training para DPT

Script para treinar a cabeça do modelo DPT em datasets de segmentação semântica supervisionado.

## Uso

```bash
python head_training.py --config <config.yaml> --save-path <output_dir> [options]
```

## Parâmetros

### Obrigatórios
- `--config`: Caminho para o arquivo de configuração YAML
- `--save-path`: Diretório para salvar checkpoints e logs

### Opcionais
- `--class-group`: Grupo de classes para treinar (definido em `config.yaml`)
  - Se omitido, treina em todas as classes
- `--pretrained`: Caminho para modelo pré-treinado
- `--resume`: Caminho para checkpoint para retomar treinamento

## Configuração (YAML)

O arquivo de configuração deve conter:

```yaml
dataset:
  name: "pascal_voc"
  root_dir: "/caminho/para/dataset"
  
train:
  epochs: 60
  batch_size: 8
  learning_rate: 0.0001
  optimizer: "AdamW"
  weight_decay: 0.01
  scheduler: "poly"
  num_workers: 4

model:
  encoder_size: "base"
  features: 256
  out_channels: [256, 512, 1024, 1024]
  use_bn: False

class_groups:
  vehicles:
    - aeroplane
    - bicycle
    - car
    - bus
```

## Dataset

O script espera arquivos de splits em:
- `{root_dir}/splits/train.txt` - Lista de pares imagem-máscara para treino
- `{root_dir}/splits/val.txt` - Lista de pares imagem-máscara para validação

Formato dos arquivos:
```
JPEGImages/img1.jpg SegmentationClass/img1.png
JPEGImages/img2.jpg SegmentationClass/img2.png
...
```

## Exemplos

### Treino completo em PASCAL VOC
```bash
python head_training.py \
    --config configs/pascal.yaml \
    --save-path training-logs/pascal_full
```

### Treino em grupo específico de classes
```bash
python head_training.py \
    --config configs/pascal.yaml \
    --save-path training-logs/pascal_vehicles \
    --class-group vehicles
```

### Retomar treinamento
```bash
python head_training.py \
    --config configs/pascal.yaml \
    --save-path training-logs/pascal_full \
    --resume training-logs/pascal_full/latest.pth
```

## Saída

O script salva:
- `latest.pth` - Último checkpoint
- `best.pth` - Melhor checkpoint (maior mIoU)
- `train.log` - Log de treinamento
- TensorBoard events em `events.out.tfevents`

## Visualizar com TensorBoard

```bash
tensorboard --logdir training-logs/pascal_full
```

## Formato de Entrada

### Imagens
- PNG, JPG, JPEG
- Qualquer tamanho (redimensionadas para 512x512)
- Normalizadas automaticamente

### Máscaras
- PNG em escala de cinza (8-bit)
- Valores de pixel = classe ID
- Valor 255 é ignorado durante treinamento

## Métricas

O script rastreia:
- **Loss**: CrossEntropyLoss
- **mIoU**: Mean Intersection over Union
- **IoU por classe**: IoU individual de cada classe
- **Learning Rate**: Taxa de aprendizado ao longo do tempo

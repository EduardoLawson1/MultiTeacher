"""
util/fusion.py

Funções de fusão das saídas multi-head do professor e do aluno.
Genérico para qualquer número de grupos de classes.
"""

import torch


def build_group_to_global(class_groups: dict, class_list: list, include_background: bool = False) -> list:
    """
    Constrói o mapeamento de índice local (dentro de cada head) para
    índice global (posição na lista completa de classes do dataset).

    Quando include_background=True, cada head foi treinada com background
    como índice local 0 adicional. O mapeamento reflete isso:
      - índice local 0 → índice global 0 (background)
      - índice local 1 → índice global da 1ª classe do grupo
      - índice local 2 → índice global da 2ª classe do grupo
      - ...

    Para o grupo 'background' especificamente (que só contém ['background']),
    o resultado é [0, 0]: ambos os índices locais mapeiam para background
    global, pois a head foi treinada com background como classe 0 e
    background novamente como classe 1 (duplicata do get_class_mapping).

    Args:
        class_groups: dict ordenado vindo do yaml.
        class_list: lista de classes do dataset ordenada por índice global.
        include_background: se True, índice local 0 de cada head é background.

    Returns:
        group_to_global: lista de tensores, um por grupo.
            len(group_to_global[i]) == group_nclass[i] sempre.
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_list)}

    group_to_global = []
    for group_name, class_names in class_groups.items():
        indices = []

        if include_background:
            # índice local 0 → background global (0)
            indices.append(0)

        for name in class_names:
            if name not in class_to_idx:
                raise ValueError(
                    f"Classe '{name}' do grupo '{group_name}' não encontrada "
                    f"na class_list do dataset. Verifique o yaml."
                )
            # inclui sempre, mesmo que seja background (índice global 0)
            # isso garante que len(indices) == group_nclass[i]
            indices.append(class_to_idx[name])

        group_to_global.append(torch.tensor(indices, dtype=torch.long))

    return group_to_global


def fuse_predictions(outs: list, group_to_global: list, nclass: int) -> torch.Tensor:
    """
    Monta o tensor completo de logits [B, nclass, H, W] a partir das
    saídas parciais de cada head, colocando cada grupo nas posições
    corretas do eixo de classes.

    Args:
        outs: lista de tensores [B, group_nclass[i], H, W], um por grupo.
        group_to_global: saída de build_group_to_global.
        nclass: número total de classes (ex: 21 para VOC).

    Returns:
        fused: tensor [B, nclass, H, W] com logits nas posições globais.
    """
    B, _, H, W = outs[0].shape
    device = outs[0].device

    fused = torch.zeros(B, nclass, H, W, device=device, dtype=outs[0].dtype)

    for out, global_indices in zip(outs, group_to_global):
        fused[:, global_indices.to(device), :, :] = out

    return fused  # [B, nclass, H, W]


def fuse_teacher_predictions(
    outs: list,
    group_to_global: list,
    nclass: int,
    conf_thresh: float = 0.0,
) -> tuple:
    """
    Gera mask_u_w e conf_u_w para o professor, escolhendo por pixel
    a head com maior confiança (softmax max) e mapeando o índice local
    vencedor para o índice global correspondente.

    Args:
        outs: lista de tensores [B, group_nclass[i], H, W], um por grupo.
        group_to_global: saída de build_group_to_global.
        nclass: número total de classes.
        conf_thresh: limiar mínimo de confiança para uma head participar
            da competição por pixel.

    Returns:
        mask_u_w: tensor [B, H, W] com índices globais (pseudo-rótulo).
        conf_u_w: tensor [B, H, W] com a confiança vencedora por pixel.
    """
    B, _, H, W = outs[0].shape
    device = outs[0].device

    best_conf = torch.full((B, H, W), fill_value=-1.0, device=device)
    best_mask = torch.zeros((B, H, W), dtype=torch.long, device=device)

    for out, global_indices in zip(outs, group_to_global):
        probs = out.softmax(dim=1)                  # [B, group_nclass, H, W]
        conf, local_idx = probs.max(dim=1)          # [B, H, W]

        global_indices_map = global_indices.to(device)
        global_idx = global_indices_map[local_idx]  # [B, H, W]

        eligible = (conf > conf_thresh) & (conf > best_conf)
        best_conf = torch.where(eligible, conf, best_conf)
        best_mask = torch.where(eligible, global_idx, best_mask)

    return best_mask, best_conf  # mask_u_w, conf_u_w
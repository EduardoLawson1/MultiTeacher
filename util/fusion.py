"""
util/fusion.py

Funções de fusão das saídas multi-head do professor e do aluno.
Genérico para qualquer número de grupos de classes.
"""

import torch


def build_group_to_global(class_groups: dict, class_list: list) -> list:
    """
    Constrói o mapeamento de índice local (dentro de cada head) para
    índice global (posição na lista completa de classes do dataset).

    Args:
        class_groups: dict ordenado vindo do yaml, ex:
            {'background': ['background'], 'vehicles': ['aeroplane', ...], ...}
        class_list: lista de classes do dataset ordenada por índice global, ex:
            ['background', 'aeroplane', 'bicycle', ..., 'tv/monitor']  (VOC)

    Returns:
        group_to_global: lista de tensores, um por grupo.
            group_to_global[i][j] = índice global da j-ésima classe do grupo i.

    Exemplo (VOC):
        group_to_global[0] = tensor([0])               # background
        group_to_global[1] = tensor([1, 2, 3, 4, 5, 6, 14])  # vehicles
        ...
    """
    # monta lookup nome -> índice global uma única vez
    class_to_idx = {name: idx for idx, name in enumerate(class_list)}

    group_to_global = []
    for group_name, class_names in class_groups.items():
        indices = []
        for name in class_names:
            if name not in class_to_idx:
                raise ValueError(
                    f"Classe '{name}' do grupo '{group_name}' não encontrada "
                    f"na class_list do dataset. Verifique o yaml."
                )
            indices.append(class_to_idx[name])
        group_to_global.append(torch.tensor(indices, dtype=torch.long))

    return group_to_global


def fuse_predictions(outs: list, group_to_global: list, nclass: int) -> torch.Tensor:
    """
    Monta o tensor completo de logits [B, nclass, H, W] a partir das
    saídas parciais de cada head, colocando cada grupo nas posições
    corretas do eixo de classes.

    Usado pelo aluno para gerar pred_x / pred_u_s1 / pred_u_s2 completos,
    que serão passados para criterion_l e criterion_u.

    Args:
        outs: lista de tensores [B, group_nclass[i], H, W], um por grupo.
        group_to_global: saída de build_group_to_global — lista de tensores
            com os índices globais de cada grupo.
        nclass: número total de classes (ex: 21 para VOC).

    Returns:
        fused: tensor [B, nclass, H, W] com logits nas posições globais.
               Posições não cobertas por nenhum grupo ficam com 0.0
               (não devem existir se class_groups cobre todas as classes).
    """
    B, _, H, W = outs[0].shape
    device = outs[0].device

    fused = torch.zeros(B, nclass, H, W, device=device, dtype=outs[0].dtype)

    for group_idx, (out, global_indices) in enumerate(zip(outs, group_to_global)):
        fused[:, global_indices.to(device), :, :] = out

    return fused  # [B, nclass, H, W]


def fuse_teacher_predictions(
    outs: list,
    group_to_global: list,
    nclass: int,
) -> tuple:
    """
    Gera mask_u_w e conf_u_w para o professor, escolhendo por pixel
    a head com maior confiança (softmax max) e mapeando o índice local
    vencedor para o índice global correspondente.

    Args:
        outs: lista de tensores [B, group_nclass[i], H, W], um por grupo.
        group_to_global: saída de build_group_to_global.
        nclass: número total de classes.

    Returns:
        mask_u_w: tensor [B, H, W] com índices globais (pseudo-rótulo).
        conf_u_w: tensor [B, H, W] com a confiança vencedora por pixel.
    """
    B, _, H, W = outs[0].shape
    device = outs[0].device

    best_conf = torch.full((B, H, W), fill_value=-1.0, device=device)
    best_mask = torch.zeros((B, H, W), dtype=torch.long, device=device)

    for out, global_indices in zip(outs, group_to_global):
        probs = out.softmax(dim=1)                    # [B, group_nclass, H, W]
        conf, local_idx = probs.max(dim=1)            # [B, H, W] cada

        # mapeia índice local -> índice global via lookup
        global_indices_map = global_indices.to(device)
        global_idx = global_indices_map[local_idx]    # [B, H, W]

        # atualiza pixels onde essa head é mais confiante
        update_mask = conf > best_conf
        best_conf = torch.where(update_mask, conf, best_conf)
        best_mask = torch.where(update_mask, global_idx, best_mask)

    return best_mask, best_conf  # mask_u_w, conf_u_w
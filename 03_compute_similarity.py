"""
03_compute_similarity.py
用法:
  python 03_compute_similarity.py \
    --src_emb embeddings/anatomy/biobert_src_emb.npy \
    --tgt_emb embeddings/anatomy/biobert_tgt_emb.npy \
    --src_entities data/parsed/mouse.json \
    --tgt_entities data/parsed/human.json \
    --out embeddings/anatomy/biobert_sim.npy \
    --index embeddings/anatomy/entity_index.json
"""
import argparse
import json
import numpy as np


def cosine_sim_matrix(src: np.ndarray, tgt: np.ndarray,
                      chunk_size: int = 512) -> np.ndarray:
    """
    计算余弦相似度矩阵。
    先L2归一化，再分块矩阵乘法，避免[M*N]一次性占用过多内存。
    src: [M, dim]
    tgt: [N, dim]
    return: [M, N]，值域[0,1]（余弦相似度经过归一化后非负）
    """
    # L2归一化
    src_norm = src / (np.linalg.norm(src, axis=1, keepdims=True) + 1e-9)
    tgt_norm = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-9)

    M = src_norm.shape[0]
    N = tgt_norm.shape[0]
    sim = np.zeros((M, N), dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        # [chunk, dim] @ [dim, N] -> [chunk, N]
        sim[start:end] = src_norm[start:end] @ tgt_norm.T

    # 余弦相似度可能因浮点误差略超[-1,1]，裁剪到[0,1]
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def build_entity_index(src_entities: list, tgt_entities: list) -> dict:
    """
    构建URI到矩阵行/列索引的映射，保存为JSON。
    格式：
    {
        "src": {"http://...uri1": 0, "http://...uri2": 1, ...},
        "tgt": {"http://...uri1": 0, ...}
    }
    """
    return {
        "src": {e["uri"]: i for i, e in enumerate(src_entities)},
        "tgt": {e["uri"]: i for i, e in enumerate(tgt_entities)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_emb",      required=True)
    parser.add_argument("--tgt_emb",      required=True)
    parser.add_argument("--src_entities", required=True)
    parser.add_argument("--tgt_entities", required=True)
    parser.add_argument("--out",          required=True, help="相似度矩阵npy输出路径")
    parser.add_argument("--index",        required=True, help="entity_index.json输出路径")
    args = parser.parse_args()

    src_emb = np.load(args.src_emb)
    tgt_emb = np.load(args.tgt_emb)
    print(f"src embedding shape: {src_emb.shape}")
    print(f"tgt embedding shape: {tgt_emb.shape}")

    sim = cosine_sim_matrix(src_emb, tgt_emb)
    print(f"相似度矩阵 shape: {sim.shape}")
    print(f"统计信息 -> min: {sim.min():.4f}, max: {sim.max():.4f}, "
          f"mean: {sim.mean():.4f}, std: {sim.std():.4f}")

    # 分布检查：如果mean>0.9或mean<0.05，说明embedding提取有问题
    if sim.mean() > 0.9:
        print("[WARNING] 相似度均值过高（>0.9），检查embedding提取方式是否正确")
    if sim.mean() < 0.05:
        print("[WARNING] 相似度均值过低（<0.05），检查文本拼接或模型加载是否正常")

    np.save(args.out, sim)
    print(f"相似度矩阵已保存: {args.out}")

    # 保存entity index
    with open(args.src_entities, encoding="utf-8") as f:
        src_entities = json.load(f)
    with open(args.tgt_entities, encoding="utf-8") as f:
        tgt_entities = json.load(f)

    index = build_entity_index(src_entities, tgt_entities)
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Entity index已保存: {args.index}")

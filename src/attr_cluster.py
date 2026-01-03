import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
import tqdm
import cupy as cp
import scipy.sparse as sp
import leidenalg
import igraph as ig
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from cupy.sparse.linalg import eigsh
from scipy.stats import t as student_t
from scipy.special import softmax

from src.utils import *
from src.community_report import community_report_batch


# 添加辅助函数：将任意整数转换为有效的 seed
def _normalize_seed(seed):
    """
    将任意整数转换为 leidenalg 可接受的 32 位有符号整数范围
    范围: -2147483648 到 2147483647
    """
    if seed is None:
        return None
    # 使用模运算将 seed 映射到有效范围
    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    # 先取绝对值的模，然后映射到有效范围
    normalized = seed % (INT32_MAX - INT32_MIN + 1) + INT32_MIN
    return int(normalized)


def compute_soft_assignment(embeddings, cluster_centers):
    """
    计算节点到cluster中心的软分配 q_iu
    使用Student's t-distribution
    
    Args:
        embeddings: (N, D) 节点嵌入
        cluster_centers: (K, D) 聚类中心嵌入
    
    Returns:
        q_iu: (N, K) 软分配矩阵
    """
    N = embeddings.shape[0]
    K = cluster_centers.shape[0]
    
    # 计算距离的平方
    distances_sq = np.zeros((N, K))
    for i in range(N):
        for u in range(K):
            diff = embeddings[i] - cluster_centers[u]
            distances_sq[i, u] = np.sum(diff ** 2)
    
    # 使用Student's t-distribution计算q_iu
    # q_iu = (1 + ||z_i - μ_u||²)^(-1) / Σ_k (1 + ||z_i - μ_k||²)^(-1)
    numerator = 1.0 / (1.0 + distances_sq)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    q_iu = numerator / denominator
    
    return q_iu


def compute_target_distribution(q_iu):
    """
    计算目标分布 p_iu
    p_iu = q_iu² / Σ_i q_iu / (Σ_k (q_ik² / Σ_i q_ik))
    
    Args:
        q_iu: (N, K) 软分配矩阵
    
    Returns:
        p_iu: (N, K) 目标分布
    """
    # q_iu² / Σ_i q_iu
    freq = np.sum(q_iu, axis=0, keepdims=True)  # (1, K)
    numerator = (q_iu ** 2) / freq  # (N, K)
    
    # 归一化
    denominator = np.sum(numerator, axis=1, keepdims=True)  # (N, 1)
    p_iu = numerator / (denominator + 1e-10)
    
    return p_iu


def compute_kl_loss(p_iu, q_iu):
    """
    计算KL散度损失
    L_c = KL(P||Q) = Σ_i Σ_u p_iu log(p_iu / q_iu)
    
    Args:
        p_iu: (N, K) 目标分布
        q_iu: (N, K) 当前分布
    
    Returns:
        loss: KL散度损失值
    """
    # 避免log(0)
    epsilon = 1e-10
    p_iu_safe = np.clip(p_iu, epsilon, 1.0)
    q_iu_safe = np.clip(q_iu, epsilon, 1.0)
    
    loss = np.sum(p_iu_safe * np.log(p_iu_safe / q_iu_safe))
    return loss


def update_cluster_centers(embeddings, q_iu):
    """
    根据软分配更新聚类中心
    μ_u = Σ_i q_iu * z_i / Σ_i q_iu
    
    Args:
        embeddings: (N, D) 节点嵌入
        q_iu: (N, K) 软分配矩阵
    
    Returns:
        new_centers: (K, D) 更新后的聚类中心
    """
    # 归一化权重
    weights = q_iu / (np.sum(q_iu, axis=0, keepdims=True) + 1e-10)  # (N, K)
    
    # 加权平均
    new_centers = np.dot(weights.T, embeddings)  # (K, D)
    
    return new_centers


def get_node_embeddings(graph, node_list):
    """
    从图中提取节点的嵌入向量
    
    Args:
        graph: igraph.Graph 对象
        node_list: 节点列表
    
    Returns:
        embeddings: (N, D) 嵌入矩阵
    """
    embeddings = []
    embedding_dim = None
    
    for node_idx in range(len(node_list)):
        # 尝试从igraph节点属性中获取嵌入
        if 'embedding' in graph.vs.attributes():
            emb = graph.vs[node_idx]['embedding']
            if emb is not None:
                if isinstance(emb, str):
                    emb = ast.literal_eval(emb)
                embeddings.append(np.array(emb))
                if embedding_dim is None:
                    embedding_dim = len(emb)
            else:
                if embedding_dim is None:
                    embedding_dim = 128  # 默认维度
                embeddings.append(np.zeros(embedding_dim))
        else:
            # 如果没有嵌入，使用随机初始化
            if embedding_dim is None:
                embedding_dim = 128
            embeddings.append(np.random.randn(embedding_dim) * 0.01)
    
    return np.array(embeddings)


def leiden_with_kl_optimization(
    ig_graph,
    node_list,
    initial_partition,
    gamma=1.0,
    max_iterations=10,
    tolerance=1e-4,
    seed=None
):
    """
    改进的Leiden算法，结合KL散度优化
    
    Args:
        ig_graph: igraph.Graph 对象
        node_list: 节点列表
        initial_partition: 初始Leiden分区
        gamma: 平衡modularity和KL散度的系数
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
        seed: 随机种子
    
    Returns:
        final_partition: 优化后的分区
        history: 优化历史记录
    """
    if seed is not None:
        np.random.seed(seed if seed >= 0 else abs(seed))
    
    # 获取节点嵌入
    embeddings = get_node_embeddings(ig_graph, node_list)
    N, D = embeddings.shape
    
    # 初始化聚类中心和分配
    current_membership = list(initial_partition.membership)
    K = len(set(current_membership))
    
    print(f"Initial number of clusters: {K}")
    print(f"Embedding dimension: {D}")
    
    # 初始化聚类中心
    cluster_centers = np.zeros((K, D))
    for k in range(K):
        cluster_nodes = [i for i, c in enumerate(current_membership) if c == k]
        if len(cluster_nodes) > 0:
            cluster_centers[k] = np.mean(embeddings[cluster_nodes], axis=0)
    
    history = {
        'modularity': [],
        'kl_loss': [],
        'total_loss': []
    }
    
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
        
        # 1. 计算软分配 q_iu
        q_iu = compute_soft_assignment(embeddings, cluster_centers)
        
        # 2. 计算目标分布 p_iu
        p_iu = compute_target_distribution(q_iu)
        
        # 3. 计算KL散度损失
        kl_loss = compute_kl_loss(p_iu, q_iu)
        
        # 4. 计算modularity
        modularity = initial_partition.quality()
        
        # 5. 总损失
        total_loss = modularity + gamma * kl_loss
        
        history['modularity'].append(modularity)
        history['kl_loss'].append(kl_loss)
        history['total_loss'].append(total_loss)
        
        print(f"Modularity: {modularity:.6f}")
        print(f"KL Loss: {kl_loss:.6f}")
        print(f"Total Loss: {total_loss:.6f}")
        
        # 6. 检查收敛
        if iteration > 0:
            loss_change = abs(history['total_loss'][-1] - history['total_loss'][-2])
            if loss_change < tolerance:
                print(f"Converged with loss change: {loss_change:.6f}")
                break
        
        # 7. 使用p_iu作为软标签更新聚类中心
        cluster_centers = update_cluster_centers(embeddings, p_iu)
        
        # 8. 基于p_iu更新hard assignment（用于下一次迭代）
        # 选择概率最大的cluster作为hard assignment
        new_membership = np.argmax(p_iu, axis=1).tolist()
        
        # 9. 使用新的membership重新优化modularity
        # 创建新的partition并进行一次Leiden优化
        try:
            # 使用新的membership初始化
            temp_partition = leidenalg.ModularityVertexPartition(
                ig_graph,
                initial_membership=new_membership,
                weights='weight' if 'weight' in ig_graph.es.attributes() else None
            )
            
            # 进行一次refinement
            optimizer = leidenalg.Optimiser()
            diff = optimizer.optimise_partition(temp_partition, n_iterations=1)
            
            # 更新partition和membership
            initial_partition = temp_partition
            current_membership = list(temp_partition.membership)
            
            # 更新K（可能会变化）
            K_new = len(set(current_membership))
            if K_new != K:
                print(f"Number of clusters changed: {K} -> {K_new}")
                K = K_new
                # 重新初始化聚类中心
                cluster_centers = np.zeros((K, D))
                for k in range(K):
                    cluster_nodes = [i for i, c in enumerate(current_membership) if c == k]
                    if len(cluster_nodes) > 0:
                        cluster_centers[k] = np.mean(embeddings[cluster_nodes], axis=0)
            
        except Exception as e:
            print(f"Warning: Failed to refine partition: {e}")
    
    return initial_partition, history


def attribute_hierarchical_clustering(
    weighted_graph: nx.graph, attributes: pd.DataFrame
):
    cluster_info, community_mapping = compute_leiden_communities(
        graph=weighted_graph,
        max_cluster_size=10,
        seed=0xDEADBEEF,
    )

    hier_tree: dict[str, str] = {}
    cluster_node_map: dict[str, list[str]] = {}
    for community_id, info in cluster_info.items():

        cluster_node_map[community_id] = info["nodes"]
        parent_community_id = str(info["parent_cluster"])
        if parent_community_id is not None:
            hier_tree[community_id] = parent_community_id

    community_level = calculate_community_levels(hier_tree)
    results_by_level: dict[int, dict[str, list[str]]] = {}

    for community_id, level in community_level.items():
        if level not in results_by_level:
            results_by_level[level] = {}
        results_by_level[level][community_id] = cluster_node_map[community_id]
    return results_by_level


def calculate_community_levels(hier_tree):
    # Initialize a dictionary to store the level of each community
    community_levels = {}

    # Function to recursively calculate the level of a community
    def calculate_level(community_id):
        # If the level is already calculated, return it
        if community_id in community_levels:
            return community_levels[community_id]

        # Find all communities that have this community_id as their parent (children nodes)
        children = [
            comm_id
            for comm_id, parent_id in hier_tree.items()
            if parent_id == community_id
        ]

        # If there are no children, it's a leaf node, so its level is 0
        if not children:
            community_levels[community_id] = 0
            return 0

        # Otherwise, calculate the level as 1 + max level of all child communities
        level = 1 + max(calculate_level(child) for child in children)

        # Store the calculated level
        community_levels[community_id] = level
        return level

    # Calculate levels for all communities, excluding None
    all_communities = set(hier_tree.keys()).union(
        set(parent_id for parent_id in hier_tree.values() if parent_id is not None)
    )
    for community_id in all_communities:
        calculate_level(community_id)

    # Before returning, remove any entry with None as a key, if it exists
    community_levels.pop("None", None)

    return community_levels


def compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph, 
    max_cluster_size: int, 
    seed=0xDEADBEEF,
    use_kl_optimization=True,
    gamma=0.5,
    kl_max_iterations=10
):
    """
    使用改进的Leiden算法（结合KL散度优化）进行层次化聚类
    
    Args:
        graph: NetworkX图
        max_cluster_size: 最大聚类大小
        seed: 随机种子
        use_kl_optimization: 是否使用KL散度优化
        gamma: KL散度的权重系数
        kl_max_iterations: KL优化的最大迭代次数
    """
    # 规范化 seed
    seed = _normalize_seed(seed)
    
    # 转换 NetworkX 图为 igraph
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 构建 igraph 图
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
    
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    
    # 添加节点的embedding属性到igraph
    for node in node_list:
        node_idx = node_to_idx[node]
        if hasattr(graph.nodes[node], '__getitem__') and 'embedding' in graph.nodes[node]:
            ig_graph.vs[node_idx]['embedding'] = graph.nodes[node]['embedding']
    
    # 第一步：使用标准Leiden获得modularity最优的初始分区
    print("Step 1: Initial Leiden clustering for modularity optimization...")
    np.random.seed(seed if seed >= 0 else abs(seed))
    initial_partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights='weight',
        n_iterations=-1,
        seed=seed
    )
    
    print(f"Initial modularity: {initial_partition.quality():.6f}")
    print(f"Initial number of communities: {len(set(initial_partition.membership))}")
    
    # 第二步：使用KL散度优化refine分区
    if use_kl_optimization:
        print("\nStep 2: Refining with KL divergence optimization...")
        optimized_partition, history = leiden_with_kl_optimization(
            ig_graph=ig_graph,
            node_list=node_list,
            initial_partition=initial_partition,
            gamma=gamma,
            max_iterations=kl_max_iterations,
            tolerance=1e-4,
            seed=seed
        )
        partition = optimized_partition
        
        # 打印优化历史
        print("\nOptimization History:")
        for i, (mod, kl, total) in enumerate(zip(
            history['modularity'], 
            history['kl_loss'], 
            history['total_loss']
        )):
            print(f"Iter {i+1}: Modularity={mod:.6f}, KL={kl:.6f}, Total={total:.6f}")
    else:
        partition = initial_partition
    
    # 构建社区映射
    community_mapping = []
    community_info: dict[str, dict] = {}
    
    # 第一层：基础聚类结果
    for node_idx, comm_id in enumerate(partition.membership):
        node = node_list[node_idx]
        community_id = str(comm_id)
        
        if community_id not in community_info:
            community_info[community_id] = {
                "level": 0,
                "nodes": [],
                "is_final_cluster": True,
                "parent_cluster": None,
            }
        community_info[community_id]["nodes"].append(node)
        
        # 创建类似 graspologic 的 partition 对象
        class Partition:
            def __init__(self, node, cluster, level, is_final, parent):
                self.node = node
                self.cluster = cluster
                self.level = level
                self.is_final_cluster = is_final
                self.parent_cluster = parent
        
        community_mapping.append(
            Partition(node, comm_id, 0, True, None)
        )
    
    # 如果需要控制社区大小，进行递归分割
    current_level = 0
    max_level = 10
    
    while current_level < max_level:
        need_split = False
        new_communities = {}
        
        for comm_id, info in list(community_info.items()):
            if not info["is_final_cluster"]:
                continue
                
            if len(info["nodes"]) <= max_cluster_size:
                new_communities[comm_id] = info
                continue
            
            # 需要分割大社区
            need_split = True
            info["is_final_cluster"] = False
            
            print(f"\nSplitting community {comm_id} with {len(info['nodes'])} nodes...")
            
            # 构建子图
            subgraph = graph.subgraph(info["nodes"])
            sub_node_list = list(subgraph.nodes())
            sub_node_to_idx = {node: idx for idx, node in enumerate(sub_node_list)}
            
            sub_edges = [(sub_node_to_idx[u], sub_node_to_idx[v]) 
                        for u, v in subgraph.edges()]
            sub_weights = [subgraph[u][v].get('weight', 1.0) 
                          for u, v in subgraph.edges()]
            
            sub_ig_graph = ig.Graph(n=len(sub_node_list), edges=sub_edges, directed=False)
            sub_ig_graph.es['weight'] = sub_weights
            
            # 添加embedding属性
            for node in sub_node_list:
                node_idx = sub_node_to_idx[node]
                if hasattr(graph.nodes[node], '__getitem__') and 'embedding' in graph.nodes[node]:
                    sub_ig_graph.vs[node_idx]['embedding'] = graph.nodes[node]['embedding']
            
            # 对子图进行聚类（同样使用KL优化）
            sub_seed = _normalize_seed(seed + current_level if seed is not None else None)
            
            # 初始Leiden
            sub_initial_partition = leidenalg.find_partition(
                sub_ig_graph,
                leidenalg.ModularityVertexPartition,
                weights='weight',
                n_iterations=-1,
                seed=sub_seed
            )
            
            # KL优化
            if use_kl_optimization:
                sub_partition, _ = leiden_with_kl_optimization(
                    ig_graph=sub_ig_graph,
                    node_list=sub_node_list,
                    initial_partition=sub_initial_partition,
                    gamma=gamma,
                    max_iterations=kl_max_iterations,
                    tolerance=1e-4,
                    seed=sub_seed
                )
            else:
                sub_partition = sub_initial_partition
            
            # 生成新的社区 ID
            max_comm_id = max([int(cid) for cid in community_info.keys()]) + 1
            
            for node_idx, sub_comm_id in enumerate(sub_partition.membership):
                node = sub_node_list[node_idx]
                new_comm_id = str(max_comm_id + sub_comm_id)
                
                if new_comm_id not in new_communities:
                    new_communities[new_comm_id] = {
                        "level": current_level + 1,
                        "nodes": [],
                        "is_final_cluster": True,
                        "parent_cluster": comm_id,
                    }
                new_communities[new_comm_id]["nodes"].append(node)
                
                # 更新 community_mapping
                for partition_obj in community_mapping:
                    if partition_obj.node == node and partition_obj.cluster == int(comm_id):
                        partition_obj.cluster = int(new_comm_id)
                        partition_obj.level = current_level + 1
                        partition_obj.parent_cluster = comm_id
        
        if not need_split:
            break
        
        # 保留未分割的社区
        for comm_id, info in community_info.items():
            if comm_id not in new_communities and not info["is_final_cluster"]:
                new_communities[comm_id] = info
        
        community_info = new_communities
        current_level += 1
    
    return community_info, community_mapping


def compute_leiden(
    graph: nx.Graph, 
    seed=0xDEADBEEF, 
    weighted=True,
    use_kl_optimization=True,
    gamma=0.5,
    kl_max_iterations=10
) -> dict[str, list[int]]:
    """
    使用改进的Leiden算法计算单层聚类
    """
    # 规范化 seed
    seed = _normalize_seed(seed)
    
    # 转换 NetworkX 图为 igraph
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 构建 igraph 图
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    
    # 如果是加权图，添加权重
    if weighted and graph.number_of_edges() > 0:
        weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
        ig_graph.es['weight'] = weights
        weight_attr = 'weight'
    else:
        weight_attr = None
    
    # 添加embedding属性
    for node in node_list:
        node_idx = node_to_idx[node]
        if hasattr(graph.nodes[node], '__getitem__') and 'embedding' in graph.nodes[node]:
            ig_graph.vs[node_idx]['embedding'] = graph.nodes[node]['embedding']
    
    # 初始Leiden聚类
    np.random.seed(seed if seed >= 0 else abs(seed))
    initial_partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=weight_attr,
        n_iterations=-1,
        seed=seed
    )
    
    # KL优化
    if use_kl_optimization:
        optimized_partition, _ = leiden_with_kl_optimization(
            ig_graph=ig_graph,
            node_list=node_list,
            initial_partition=initial_partition,
            gamma=gamma,
            max_iterations=kl_max_iterations,
            tolerance=1e-4,
            seed=seed
        )
        partition = optimized_partition
    else:
        partition = initial_partition
    
    # 构建社区-节点映射
    c_n_mapping: dict[str, list[int]] = {}
    
    for node_idx, comm_id in enumerate(partition.membership):
        node = node_list[node_idx]
        community_id = str(comm_id)
        
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)
    
    return c_n_mapping


def compute_leiden_max_size(
    graph: nx.Graph, 
    max_cluster_size: int, 
    seed=0xDEADBEEF, 
    weighted=True,
    use_kl_optimization=True,
    gamma=0.5,
    kl_max_iterations=10
):
    """
    使用改进的Leiden算法计算聚类，并限制最大社区大小
    """
    # 规范化 seed
    seed = _normalize_seed(seed)
    
    # 转换 NetworkX 图为 igraph
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # 构建 igraph 图
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
    
    # 如果是加权图，添加权重
    if weighted and graph.number_of_edges() > 0:
        weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
        ig_graph.es['weight'] = weights
        weight_attr = 'weight'
    else:
        weight_attr = None
    
    # 添加embedding属性
    for node in node_list:
        node_idx = node_to_idx[node]
        if hasattr(graph.nodes[node], '__getitem__') and 'embedding' in graph.nodes[node]:
            ig_graph.vs[node_idx]['embedding'] = graph.nodes[node]['embedding']
    
    # 初始聚类
    np.random.seed(seed if seed >= 0 else abs(seed))
    initial_partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=weight_attr,
        n_iterations=-1,
        seed=seed
    )
    
    # KL优化
    if use_kl_optimization:
        print("Applying KL optimization to initial partition...")
        optimized_partition, _ = leiden_with_kl_optimization(
            ig_graph=ig_graph,
            node_list=node_list,
            initial_partition=initial_partition,
            gamma=gamma,
            max_iterations=kl_max_iterations,
            tolerance=1e-4,
            seed=seed
        )
        partition = optimized_partition
    else:
        partition = initial_partition
    
    # 构建初始社区-节点映射
    c_n_mapping: dict[str, list[int]] = {}
    for node_idx, comm_id in enumerate(partition.membership):
        node = node_list[node_idx]
        community_id = str(comm_id)
        
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)
    
    # 递归分割过大的社区
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        need_split = False
        new_c_n_mapping = {}
        next_comm_id = max([int(cid) for cid in c_n_mapping.keys()]) + 1
        
        for comm_id, nodes in c_n_mapping.items():
            if len(nodes) <= max_cluster_size:
                new_c_n_mapping[comm_id] = nodes
            else:
                need_split = True
                
                print(f"\nSplitting community {comm_id} with {len(nodes)} nodes (iteration {iteration+1})...")
                
                # 构建子图
                subgraph = graph.subgraph(nodes)
                sub_node_list = list(subgraph.nodes())
                sub_node_to_idx = {node: idx for idx, node in enumerate(sub_node_list)}
                
                sub_edges = [(sub_node_to_idx[u], sub_node_to_idx[v]) 
                           for u, v in subgraph.edges()]
                sub_ig_graph = ig.Graph(n=len(sub_node_list), edges=sub_edges, directed=False)
                
                if weighted and subgraph.number_of_edges() > 0:
                    sub_weights = [subgraph[u][v].get('weight', 1.0) 
                                 for u, v in subgraph.edges()]
                    sub_ig_graph.es['weight'] = sub_weights
                    sub_weight_attr = 'weight'
                else:
                    sub_weight_attr = None
                
                # 添加embedding
                for node in sub_node_list:
                    node_idx = sub_node_to_idx[node]
                    if hasattr(graph.nodes[node], '__getitem__') and 'embedding' in graph.nodes[node]:
                        sub_ig_graph.vs[node_idx]['embedding'] = graph.nodes[node]['embedding']
                
                # 对子图聚类
                sub_seed = _normalize_seed(seed + iteration if seed is not None else None)
                sub_initial_partition = leidenalg.find_partition(
                    sub_ig_graph,
                    leidenalg.ModularityVertexPartition,
                    weights=sub_weight_attr,
                    n_iterations=-1,
                    seed=sub_seed
                )
                
                # KL优化子图
                if use_kl_optimization:
                    sub_partition, _ = leiden_with_kl_optimization(
                        ig_graph=sub_ig_graph,
                        node_list=sub_node_list,
                        initial_partition=sub_initial_partition,
                        gamma=gamma,
                        max_iterations=kl_max_iterations,
                        tolerance=1e-4,
                        seed=sub_seed
                    )
                else:
                    sub_partition = sub_initial_partition
                
                # 将子社区添加到映射中
                for node_idx, sub_comm_id in enumerate(sub_partition.membership):
                    node = sub_node_list[node_idx]
                    new_comm_id_str = str(next_comm_id + sub_comm_id)
                    
                    if new_comm_id_str not in new_c_n_mapping:
                        new_c_n_mapping[new_comm_id_str] = []
                    new_c_n_mapping[new_comm_id_str].append(node)
                
                next_comm_id += len(set(sub_partition.membership))
        
        if not need_split:
            break
        
        c_n_mapping = new_c_n_mapping
        iteration += 1
    
    return c_n_mapping


def spectralClustering(graph: nx.graph, seed, l, is_weighted):
    c_n_mapping: dict[str, list[int]] = {}
    # 转换成sklearn中SpectralClustering的输入格式
    num_nodes = len(graph.nodes())

    index = np.array([node for node in graph.nodes()])
    # 谱聚类

    num_worker_spec = 32
    if not is_weighted:
        # 非加权图
        adj_matrix = nx.to_scipy_sparse_array(graph, dtype=np.int32, format="csr")
        adj_matrix.indices = adj_matrix.indices.astype(np.int32, casting="same_kind")
        adj_matrix.indptr = adj_matrix.indptr.astype(np.int32, casting="same_kind")
        sc = SpectralClustering(
            affinity="precomputed",
            assign_labels="discretize",
            random_state=seed,
            n_clusters=l,
            n_jobs=num_worker_spec,
            verbose=True,
        )
    else:
        # 加权图
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for node in tqdm.tqdm(graph.adj):
            indice_node = np.where(index == node)[0][0]
            for neighbor in graph.adj[node]:
                indice_neighbor = np.where(index == neighbor)[0][0]
                adj_matrix[indice_node][indice_neighbor] = graph.adj[node][neighbor][
                    "weight"
                ]

        adj_matrix = csr_matrix(adj_matrix)
        sc = SpectralClustering(
            affinity="precomputed",
            assign_labels="discretize",
            random_state=seed,
            n_clusters=l,
            n_jobs=num_worker_spec,
            verbose=True,
        )

    print("Start fit predict")

    # 获取聚类结果
    cluster_result = sc.fit_predict(adj_matrix)

    # 转换成c_n_mapping的格式
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping


def spectral_clustering_cupy(graph: nx.graph, seed, number_cluster, is_weighted):
    if number_cluster < 1:
        number_cluster = 1

    c_n_mapping: dict[str, list[int]] = {}
    index = np.array([node for node in graph.nodes()])
    if is_weighted:
        # 如果是加权图，使用边的权重
        adj_matrix = nx.adjacency_matrix(graph, weight="weight")
    else:
        # 如果是无权图，权重为1
        adj_matrix = nx.adjacency_matrix(graph)

    # transform the adjacency to sparse matrix
    adj_matrix = adj_matrix.astype(float)
    adj_matrix = sp.csr_matrix(adj_matrix)

    print("finish compute_laplacian_matrix")

    # 将邻接矩阵转换为CuPy的稀疏矩阵
    adj_matrix_gpu = cp.sparse.csr_matrix(adj_matrix)

    # 计算度矩阵D
    degrees = cp.array(adj_matrix_gpu.sum(axis=1)).flatten()  # 行和作为度数
    degree_matrix = cp.sparse.diags(degrees)

    # 计算拉普拉斯矩阵L = D - A
    laplacian_matrix = degree_matrix - adj_matrix_gpu

    top_k_eig = min(number_cluster, 200)
    try:
        # 使用CuPy计算拉普拉斯矩阵的前k个特征值和特征向量
        eigvals, eigvecs = eigsh(laplacian_matrix, k=top_k_eig, which="LA")
    except Exception as e:
        print(f"Error during eigendecomposition: {e}")
    finally:
        # release cuda memory
        cp.get_default_memory_pool().free_all_blocks()
    print("finish cp.linalg.eigh, number of eigvals:", len(eigvals))
    print("eigvecs shape:", eigvecs.shape)

    # 选择前k个最小特征值对应的特征向量
    eigvecs_selected = eigvecs[:, :top_k_eig]

    # 对特征向量进行k-means聚类
    kmeans = KMeans(n_clusters=number_cluster, random_state=seed)
    clusters = kmeans.fit_predict(eigvecs_selected.get())
        
    for node, label in enumerate(clusters):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])

    print("number of clusters:", len(set(c_n_mapping)))
    return c_n_mapping


def community_id_node_resize(
    c_n_mapping: dict[str, list[str]], community_df: pd.DataFrame
):
    if not community_df.empty:
        # 先将 community_id 字段转换为数值类型，遇到非数值的情况使用 NaN，然后忽略 NaN 值
        community_df["community_id_numeric"] = pd.to_numeric(
            community_df["community_id"], errors="coerce"
        )

        # 找到 community_id 中的最大值
        cur_max_id = community_df["community_id_numeric"].max() + 1
    else:
        cur_max_id = 0

    community_df["community_nodes"] = community_df["community_nodes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 创建一个新的字典，存放更新后的 community_id 和对应的社区节点
    updated_c_n_mapping = {}
    c_c_mapping = {}

    # 遍历 c_n_mapping 中的每个社区
    for _, node_list in c_n_mapping.items():
        # 获取 node_list 中每个 node 对应的 title 列表
        community_nodes = []
        for community_id in node_list:
            ct_nodes = community_df.loc[
                community_df["community_id"] == community_id, "community_nodes"
            ]
            if not ct_nodes.empty:
                # ct_nodes 是一个 Series，取第一个值
                community_nodes_list = ct_nodes.iloc[0]

                if isinstance(community_nodes_list, list):
                    community_nodes.extend(community_nodes_list)
                else:
                    print(
                        f"Warning: {community_id} does not have a list of community_nodes"
                    )
            else:
                print(f"Warning: {community_id} not found in community_df")

        # 去重处理
        unique_nodes = list(set(community_nodes))

        updated_c_n_mapping[str(cur_max_id)] = unique_nodes
        c_c_mapping[str(cur_max_id)] = node_list

        # 增加 cur_max_id，为下一个社区准备新的编号
        cur_max_id += 1

    return updated_c_n_mapping, c_c_mapping


def reconstruct_graph(community_df, final_relationships):
    graph = nx.Graph()

    node_community_map = {}

    # add nodes
    for idx, row in community_df.iterrows():
        # Only add the 'embedding' attribute to the graph
        embedding = row["embedding"] if "embedding" in row else None
        if isinstance(embedding, str):
            try:
                embedding = ast.literal_eval(embedding)
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Unable to parse embedding for {row['title']} - {row['community_id']}"
                )

        graph.add_node(row["community_id"], embedding=embedding)

        community_nodes = row["community_nodes"]
        if isinstance(community_nodes, str):
            try:
                community_nodes = ast.literal_eval(community_nodes)
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Unable to parse community_nodes for {row['title']} - {row['community_id']}"
                )
                community_nodes = []

        for nodes in community_nodes:
            node_community_map[nodes] = row["community_id"]

    for _, row in final_relationships.iterrows():
        if row["head_id"] in node_community_map:
            source = node_community_map[row["head_id"]]
        else:
            continue

        if row["tail_id"] in node_community_map:
            target = node_community_map[row["tail_id"]]
        else:
            continue

        if source == target:
            continue

            # If the edge already exists, increment the weight; otherwise, add the edge with weight 1
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)

            # 检查边中是否有 NaN
    nan_edges = [
        (u, v)
        for u, v in graph.edges
        if u is None
        or v is None
        or (isinstance(u, float) and math.isnan(u))
        or (isinstance(v, float) and math.isnan(v))
    ]

    if nan_edges:
        print(f"Detected NaN edges: {nan_edges}")
    else:
        print("No NaN edges detected.")
        # 检查节点中是否有 NaN
    nan_nodes = [
        node
        for node in graph.nodes
        if node is None or (isinstance(node, float) and math.isnan(node))
    ]

    if nan_nodes:
        print(f"Detected NaN nodes: {nan_nodes}")
    else:
        print("No NaN nodes detected.")

    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        graph.remove_edges_from(self_loops)

    return graph, community_df


def attr_cluster(
    init_graph: nx.Graph,
    final_entities,
    final_relationships,
    args,
    max_level=4,
    min_clusters=5,
):
    level = 1
    graph = init_graph
    community_df = pd.DataFrame()
    all_token = 0
    
    while level <= max_level:
        print(f"\n{'='*60}")
        print(f"Start clustering for level {level}")
        print(f"{'='*60}")

        # 1. augment graph and compute weight
        if args.augment_graph is True:
            # 计算余弦距离图
            cos_graph = compute_distance(
                graph,
                x_percentile=args.wx_weight,
                search_k=args.search_k,
                m_du_sacle=args.m_du_scale,
            )
        else:
            cos_graph = graph

        # 2. clustering with KL optimization
        if args.augment_graph is True:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, 
                    args.max_cluster_size, 
                    args.seed,
                    use_kl_optimization=True,
                    gamma=args.kl_gamma if hasattr(args, 'kl_gamma') else 0.5,
                    kl_max_iterations=args.kl_iterations if hasattr(args, 'kl_iterations') else 10
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, True
                )
        else:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, 
                    args.max_cluster_size, 
                    args.seed, 
                    False,
                    use_kl_optimization=True,
                    gamma=args.kl_gamma if hasattr(args, 'kl_gamma') else 0.5,
                    kl_max_iterations=args.kl_iterations if hasattr(args, 'kl_iterations') else 10
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, False
                )

        # check for finish
        number_of_clusters = len(c_n_mapping)
        print(f"\nNumber of clusters at level {level}: {number_of_clusters}")
        
        if number_of_clusters < min_clusters:
            print(f"Stopping: number of clusters ({number_of_clusters}) < min_clusters ({min_clusters})")
            break

        # 如果不是第一层，需要调整 community_id
        if level > 1:
            updated_c_n_mapping, c_c_mapping = community_id_node_resize(
                c_n_mapping=c_n_mapping, community_df=community_df
            )
        else:
            updated_c_n_mapping = c_n_mapping
            c_c_mapping = {}

        print(f"Number of communities: {len(updated_c_n_mapping)}")
        c_id_list = list(updated_c_n_mapping.keys())
        print(f"Community id list: {c_id_list[:10]}{'...' if len(c_id_list) > 10 else ''}")

        # 构建 level_dict，记录每个社区对应的 level
        level_dict = {
            community_id: level for community_id in updated_c_n_mapping.keys()
        }

        tmp_comunity_df_result = os.path.join(
            args.output_dir, f"tmp_community_df_{level}.csv"
        )

        tmp_comunity_df_error = os.path.join(
            args.output_dir, f"tmp_community_df_{level}_error.csv"
        )

        if os.path.exists(tmp_comunity_df_result):
            print(
                f"File {tmp_comunity_df_result} already exists. Loading existing data."
            )
            new_community_df = pd.read_csv(tmp_comunity_df_result)
            new_community_df["embedding"] = new_community_df["embedding"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            new_community_df["community_nodes"] = new_community_df[
                "community_nodes"
            ].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            print("Generating new community report.")
            new_community_df, cur_token = community_report_batch(
                communities=updated_c_n_mapping,
                c_c_mapping=c_c_mapping,
                final_entities=final_entities,
                final_relationships=final_relationships,
                exist_community_df=community_df,
                level_dict=level_dict,
                error_save_path=tmp_comunity_df_error,
                args=args,
            )
            all_token += cur_token
            print(f"Token usage for level {level}: {cur_token}")
            print(f"Total token usage so far: {all_token}")

        # update
        graph, new_community_df = reconstruct_graph(
            new_community_df, final_relationships
        )
        new_community_df.to_csv(tmp_comunity_df_result, index=False)
        community_df = pd.concat([community_df, new_community_df], ignore_index=True)
        level += 1

    print(f"\n{'='*60}")
    print(f"Clustering completed at level {level-1}")
    print(f"Total token usage: {all_token}")
    print(f"{'='*60}\n")
    
    return community_df, all_token


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'kl_gamma'):
        args.kl_gamma = 0.5  
    if not hasattr(args, 'kl_iterations'):
        args.kl_iterations = 10  
    
    print_args(args)

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    print("Graph loaded.")
    print(args.base_path)
    community_df, all_token = attr_cluster(
        init_graph=graph,
        final_entities=final_entities,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )

    output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    community_df.to_csv(output_path, index=False)
    print(f"Community report saved to {output_path}")
    print(f"Total token usage: {all_token}")
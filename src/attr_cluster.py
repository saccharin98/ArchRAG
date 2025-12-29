import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
import tqdm
import cupy as cp
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from cupy.sparse.linalg import eigsh
from collections import defaultdict
import random


from src.utils import *
from src.community_report import community_report_batch


class ManualLeiden:
    """手动实现的 Leiden 算法"""
    
    def __init__(self, graph: nx.Graph, is_weighted: bool = True, random_seed: int = 0xDEADBEEF):
        self.graph = graph
        self.is_weighted = is_weighted
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化每个节点为独立的社区
        self.node_to_community = {node: i for i, node in enumerate(graph.nodes())}
        self.community_to_nodes = {i: [node] for i, node in enumerate(graph.nodes())}
        
    def _modularity_gain(self, node, target_community):
        """计算将节点移动到目标社区的模块度增益"""
        if self.is_weighted:
            # 计算节点到目标社区的边权重之和
            edge_weight_to_community = 0
            for neighbor in self.graph.neighbors(node):
                if self.node_to_community[neighbor] == target_community:
                    edge_weight_to_community += self.graph[node][neighbor].get('weight', 1)
            
            # 计算节点的总度数
            node_degree = sum(self.graph[node][neighbor].get('weight', 1) 
                            for neighbor in self.graph.neighbors(node))
            
            # 计算目标社区的总度数
            community_degree = sum(
                sum(self.graph[n][neighbor].get('weight', 1) for neighbor in self.graph.neighbors(n))
                for n in self.community_to_nodes[target_community]
            )
            
            # 总边权重
            total_weight = sum(data.get('weight', 1) for _, _, data in self.graph.edges(data=True))
            
        else:
            # 无权图的情况
            edge_weight_to_community = sum(
                1 for neighbor in self.graph.neighbors(node)
                if self.node_to_community[neighbor] == target_community
            )
            
            node_degree = self.graph.degree(node)
            community_degree = sum(
                self.graph.degree(n) for n in self.community_to_nodes[target_community]
            )
            total_weight = self.graph.number_of_edges()
        
        if total_weight == 0:
            return 0
        
        # 模块度增益计算
        gain = edge_weight_to_community - (node_degree * community_degree) / (2 * total_weight)
        return gain
    
    def _move_nodes_phase(self):
        """第一阶段：移动节点到最优社区"""
        improvement = False
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)
        
        for node in nodes:
            current_community = self.node_to_community[node]
            
            # 找到邻居所在的社区
            neighbor_communities = set()
            for neighbor in self.graph.neighbors(node):
                neighbor_communities.add(self.node_to_community[neighbor])
            
            # 计算移动到每个邻居社区的增益
            best_community = current_community
            best_gain = 0
            
            for community in neighbor_communities:
                if community == current_community:
                    continue
                    
                gain = self._modularity_gain(node, community)
                if gain > best_gain:
                    best_gain = gain
                    best_community = community
            
            # 如果找到更好的社区，移动节点
            if best_community != current_community:
                improvement = True
                self.community_to_nodes[current_community].remove(node)
                if not self.community_to_nodes[current_community]:
                    del self.community_to_nodes[current_community]
                
                if best_community not in self.community_to_nodes:
                    self.community_to_nodes[best_community] = []
                self.community_to_nodes[best_community].append(node)
                self.node_to_community[node] = best_community
        
        return improvement
    
    def _refine_partition(self):
        """第二阶段：细化分区"""
        new_community_id = max(self.community_to_nodes.keys()) + 1 if self.community_to_nodes else 0
        
        for community_id in list(self.community_to_nodes.keys()):
            nodes = self.community_to_nodes[community_id]
            if len(nodes) <= 1:
                continue
            
            # 为社区内的节点创建子图
            subgraph = self.graph.subgraph(nodes)
            
            # 在子图中再次运行移动节点阶段
            sub_node_to_community = {node: community_id for node in nodes}
            
            nodes_shuffled = list(nodes)
            random.shuffle(nodes_shuffled)
            
            for node in nodes_shuffled:
                neighbor_communities = set()
                for neighbor in subgraph.neighbors(node):
                    neighbor_communities.add(sub_node_to_community[neighbor])
                
                best_community = sub_node_to_community[node]
                best_gain = 0
                
                for target_comm in neighbor_communities:
                    if target_comm == sub_node_to_community[node]:
                        continue
                    
                    gain = self._modularity_gain(node, target_comm)
                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_comm
                
                # 如果节点需要分离到新社区
                if best_community != community_id and best_gain > 0:
                    if best_community not in self.community_to_nodes:
                        self.community_to_nodes[new_community_id] = []
                        self.community_to_nodes[new_community_id].append(node)
                        self.node_to_community[node] = new_community_id
                        sub_node_to_community[node] = new_community_id
                        new_community_id += 1
                    else:
                        self.community_to_nodes[best_community].append(node)
                        self.node_to_community[node] = best_community
                        sub_node_to_community[node] = best_community
                    
                    self.community_to_nodes[community_id].remove(node)
    
    def _aggregate_graph(self):
        """第三阶段：聚合图"""
        new_graph = nx.Graph()
        
        # 为每个社区创建一个新节点
        for community_id in self.community_to_nodes.keys():
            new_graph.add_node(community_id)
        
        # 创建社区间的边
        edge_weights = defaultdict(float)
        
        for u, v, data in self.graph.edges(data=True):
            comm_u = self.node_to_community[u]
            comm_v = self.node_to_community[v]
            
            if comm_u != comm_v:
                edge_key = tuple(sorted([comm_u, comm_v]))
                weight = data.get('weight', 1) if self.is_weighted else 1
                edge_weights[edge_key] += weight
        
        for (comm_u, comm_v), weight in edge_weights.items():
            new_graph.add_edge(comm_u, comm_v, weight=weight)
        
        return new_graph
    
    def run(self, max_iterations: int = 100):
        """运行 Leiden 算法"""
        for iteration in range(max_iterations):
            # 第一阶段:移动节点
            improved = self._move_nodes_phase()
            
            if not improved:
                break
            
            # 第二阶段:细化分区
            self._refine_partition()
            
            # 第三阶段:聚合图(为下一次迭代准备)
            if iteration < max_iterations - 1:
                self.graph = self._aggregate_graph()
                
                # 重新初始化节点到社区的映射
                # 聚合后的图中,每个节点就是一个社区ID
                self.node_to_community = {node: node for node in self.graph.nodes()}
                self.community_to_nodes = {node: [node] for node in self.graph.nodes()}
        
        return self.node_to_community


class ManualHierarchicalLeiden:
    """手动实现的分层 Leiden 算法"""
    
    def __init__(self, graph: nx.Graph, max_cluster_size: int, is_weighted: bool = True, 
                 random_seed: int = 0xDEADBEEF):
        self.graph = graph
        self.max_cluster_size = max_cluster_size
        self.is_weighted = is_weighted
        self.random_seed = random_seed
        self.partitions = []
        
    def run(self):
        """运行分层 Leiden 算法"""
        current_graph = self.graph.copy()
        # 使用字典映射当前图节点到原始图节点
        current_to_original = {node: [node] for node in self.graph.nodes()}
        level = 0
        parent_mapping = {node: None for node in self.graph.nodes()}
        
        while True:
            # 运行 Leiden 算法
            leiden = ManualLeiden(current_graph, self.is_weighted, self.random_seed)
            node_to_community = leiden.run()
            
            # 检查是否所有社区都满足大小限制
            community_to_nodes = defaultdict(list)
            for node, community in node_to_community.items():
                # 获取原始节点
                original_nodes = current_to_original[node]
                community_to_nodes[community].extend(original_nodes)
            
            all_clusters_valid = all(
                len(nodes) <= self.max_cluster_size 
                for nodes in community_to_nodes.values()
            )
            
            # 记录当前层级的分区
            for community_id, nodes in community_to_nodes.items():
                for node in nodes:
                    cluster_id = f"{level}_{community_id}"
                    partition = {
                        'node': node,
                        'cluster': cluster_id,
                        'level': level,
                        'parent_cluster': parent_mapping.get(node),
                        'is_final_cluster': all_clusters_valid
                    }
                    self.partitions.append(partition)
            
            # 如果所有社区都满足大小限制，停止
            if all_clusters_valid:
                break
            
            # 否则，为需要进一步分割的社区创建新层级
            new_graph = nx.Graph()
            new_to_original = {}  # 映射新图节点到原始节点
            new_parent_mapping = {}
            
            for community_id, nodes in community_to_nodes.items():
                if len(nodes) > self.max_cluster_size:
                    # 创建子图（使用原始节点）
                    subgraph = self.graph.subgraph(nodes)
                    
                    # 为子图中的每个节点创建映射
                    for node in subgraph.nodes():
                        new_graph.add_node(node)
                        new_to_original[node] = [node]  # 保持原始节点
                        new_parent_mapping[node] = f"{level}_{community_id}"
                    
                    # 添加边
                    for u, v, data in subgraph.edges(data=True):
                        new_graph.add_edge(u, v, **data)
            
            if new_graph.number_of_nodes() == 0:
                break
            
            current_graph = new_graph
            current_to_original = new_to_original  # 更新映射
            parent_mapping = new_parent_mapping
            level += 1
            
            # 防止无限循环
            if level > 10:
                break
        
        return self.partitions


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
    graph: nx.Graph | nx.DiGraph, max_cluster_size: int, seed=0xDEADBEEF
):
    """使用手动实现的分层 Leiden 算法"""
    hierarchical_leiden = ManualHierarchicalLeiden(
        graph, max_cluster_size=max_cluster_size, is_weighted=True, random_seed=seed
    )
    community_mapping = hierarchical_leiden.run()
    
    community_info: dict[str, dict] = {}

    for partition in community_mapping:
        community_id = str(partition['cluster'])
        if community_id not in community_info:
            community_info[community_id] = {
                "level": partition['level'],
                "nodes": [],
                "is_final_cluster": partition['is_final_cluster'],
                "parent_cluster": partition['parent_cluster'],
            }
        community_info[community_id]["nodes"].append(partition['node'])
    
    return community_info, community_mapping


def compute_leiden(
    graph: nx.Graph, seed=0xDEADBEEF, weighted=True
) -> dict[str, list[int]]:
    """使用手动实现的 Leiden 算法计算一层"""
    leiden = ManualLeiden(graph, is_weighted=weighted, random_seed=seed)
    node_to_community = leiden.run()
    
    c_n_mapping: dict[str, list[int]] = {}

    for node, community in node_to_community.items():
        community_id = str(community)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)

    return c_n_mapping


def compute_leiden_max_size(
    graph: nx.Graph, max_cluster_size: int, seed=0xDEADBEEF, weighted=True
):
    """使用手动实现的分层 Leiden 算法，限制最大社区大小"""
    hierarchical_leiden = ManualHierarchicalLeiden(
        graph, max_cluster_size=max_cluster_size, is_weighted=weighted, random_seed=seed
    )
    community_mapping = hierarchical_leiden.run()
    
    c_n_mapping: dict[str, list[int]] = {}

    for partition in community_mapping:
        if not partition['is_final_cluster']:
            continue
        community_id = str(partition['cluster'])
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(partition['node'])

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
        print(f"Start clustering for level {level}")

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

        # 2. clustering
        if args.augment_graph is True:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, args.max_cluster_size, args.seed
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, True
                )
        else:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, args.max_cluster_size, args.seed, False
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, False
                )

        # check for finish
        number_of_clusters = len(c_n_mapping)
        if number_of_clusters < min_clusters:
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
        print(f"Community id list: {c_id_list}")

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
            
            # 确保 level 列存在
            if "level" not in new_community_df.columns:
                # 从 community_id 中提取 level 或使用 level_dict
                if new_community_df["community_id"].iloc[0] in level_dict:
                    new_community_df["level"] = new_community_df["community_id"].map(level_dict)
                else:
                    new_community_df["level"] = level
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
            print(f"cur token usage for current level: {cur_token}")
            
            # 确保生成的 DataFrame 包含 level 列
            if "level" not in new_community_df.columns:
                new_community_df["level"] = new_community_df["community_id"].map(level_dict)

        # update
        graph, new_community_df = reconstruct_graph(
            new_community_df, final_relationships
        )
        new_community_df.to_csv(tmp_comunity_df_result, index=False)
        community_df = pd.concat([community_df, new_community_df], ignore_index=True)
        level += 1

    return community_df, all_token


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    print_args(args)

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
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
import os
import ast
import numpy as np
import networkx as nx
import pandas as pd
from graspologic.partition import hierarchical_leiden, leiden

# 调试脚本：打印Leiden算法返回值的详细信息

def debug_leiden_outputs(graph: nx.Graph, max_cluster_size: int = 10, seed=0xDEADBEEF):
    """
    打印Leiden算法各个函数的返回值结构
    """
    print("=" * 80)
    print("1. 测试 compute_leiden() - 单层Leiden")
    print("=" * 80)
    
    # 单层Leiden
    community_mapping = leiden(graph, is_weighted=True, random_seed=seed)
    
    print(f"\n类型: {type(community_mapping)}")
    print(f"长度: {len(community_mapping)}")
    print(f"\n前5个元素:")
    for i, (node, community) in enumerate(list(community_mapping.items())[:5]):
        print(f"  [{i}] node={node} (type: {type(node)}), community={community} (type: {type(community)})")
    
    # 构建c_n_mapping
    c_n_mapping = {}
    for node, community in community_mapping.items():
        community_id = str(community)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)
    
    print(f"\n转换后的 c_n_mapping:")
    print(f"  类型: {type(c_n_mapping)}")
    print(f"  社区数量: {len(c_n_mapping)}")
    print(f"  键的类型: {type(list(c_n_mapping.keys())[0])}")
    print(f"  值的类型: {type(list(c_n_mapping.values())[0])}")
    print(f"\n前3个社区:")
    for i, (comm_id, nodes) in enumerate(list(c_n_mapping.items())[:3]):
        print(f"  社区 '{comm_id}': {len(nodes)} 个节点")
        print(f"    前5个节点: {nodes[:5]}")
        print(f"    节点类型: {type(nodes[0])}")
    
    print("\n" + "=" * 80)
    print("2. 测试 compute_leiden_max_size() - 分层Leiden")
    print("=" * 80)
    
    # 分层Leiden
    community_mapping_hier = hierarchical_leiden(
        graph, 
        max_cluster_size=max_cluster_size, 
        random_seed=seed, 
        is_weighted=True
    )
    
    print(f"\n类型: {type(community_mapping_hier)}")
    print(f"长度: {len(community_mapping_hier)}")
    print(f"\n前5个partition对象:")
    for i, partition in enumerate(list(community_mapping_hier)[:5]):
        print(f"\n  [{i}] Partition对象:")
        print(f"    类型: {type(partition)}")
        print(f"    属性: {dir(partition)}")
        print(f"    - node: {partition.node} (type: {type(partition.node)})")
        print(f"    - cluster: {partition.cluster} (type: {type(partition.cluster)})")
        print(f"    - parent_cluster: {partition.parent_cluster} (type: {type(partition.parent_cluster)})")
        print(f"    - level: {partition.level} (type: {type(partition.level)})")
        print(f"    - is_final_cluster: {partition.is_final_cluster} (type: {type(partition.is_final_cluster)})")
    
    # 构建c_n_mapping (只保留final clusters)
    c_n_mapping_hier = {}
    for partition in community_mapping_hier:
        if not partition.is_final_cluster:
            continue
        community_id = str(partition.cluster)
        if community_id not in c_n_mapping_hier:
            c_n_mapping_hier[community_id] = []
        c_n_mapping_hier[community_id].append(partition.node)
    
    print(f"\n转换后的 c_n_mapping (仅final clusters):")
    print(f"  社区数量: {len(c_n_mapping_hier)}")
    print(f"\n前3个社区:")
    for i, (comm_id, nodes) in enumerate(list(c_n_mapping_hier.items())[:3]):
        print(f"  社区 '{comm_id}': {len(nodes)} 个节点")
        print(f"    前5个节点: {nodes[:5]}")
    
    print("\n" + "=" * 80)
    print("3. 测试 compute_leiden_communities() - 完整社区信息")
    print("=" * 80)
    
    community_info = {}
    for partition in community_mapping_hier:
        community_id = str(partition.cluster)
        if community_id not in community_info:
            community_info[community_id] = {
                "level": partition.level,
                "nodes": [],
                "is_final_cluster": partition.is_final_cluster,
                "parent_cluster": partition.parent_cluster,
            }
        community_info[community_id]["nodes"].append(partition.node)
    
    print(f"\ncommunity_info 结构:")
    print(f"  类型: {type(community_info)}")
    print(f"  社区数量: {len(community_info)}")
    print(f"\n前3个社区的详细信息:")
    for i, (comm_id, info) in enumerate(list(community_info.items())[:3]):
        print(f"\n  社区 '{comm_id}':")
        print(f"    level: {info['level']} (type: {type(info['level'])})")
        print(f"    is_final_cluster: {info['is_final_cluster']} (type: {type(info['is_final_cluster'])})")
        print(f"    parent_cluster: {info['parent_cluster']} (type: {type(info['parent_cluster'])})")
        print(f"    nodes: {len(info['nodes'])} 个节点, 前5个: {info['nodes'][:5]}")
    
    print("\n" + "=" * 80)
    print("4. 层级关系分析")
    print("=" * 80)
    
    # 构建层级树
    hier_tree = {}
    for community_id, info in community_info.items():
        parent_community_id = str(info["parent_cluster"])
        if parent_community_id != "None":
            hier_tree[community_id] = parent_community_id
    
    print(f"\n层级树 (hier_tree):")
    print(f"  类型: {type(hier_tree)}")
    print(f"  条目数: {len(hier_tree)}")
    print(f"\n前10个父子关系:")
    for i, (child, parent) in enumerate(list(hier_tree.items())[:10]):
        print(f"  子社区 '{child}' -> 父社区 '{parent}'")
    
    # 统计每个level的社区数
    level_counts = {}
    for info in community_info.values():
        level = info['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"\n各层级的社区数量:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} 个社区")
    
    # 统计final clusters的level分布
    final_level_counts = {}
    for info in community_info.values():
        if info['is_final_cluster']:
            level = info['level']
            final_level_counts[level] = final_level_counts.get(level, 0) + 1
    
    print(f"\nFinal clusters的层级分布:")
    for level in sorted(final_level_counts.keys()):
        print(f"  Level {level}: {final_level_counts[level]} 个final社区")
    
    print("\n" + "=" * 80)
    print("5. 关键返回值总结")
    print("=" * 80)
    
    print("""
你需要实现的返回值格式:

1. compute_leiden(graph, seed, weighted) -> dict[str, list[int]]
   返回: {"0": [node1, node2], "1": [node3, node4], ...}
   - 键: 字符串类型的社区ID
   - 值: 节点列表
   
2. compute_leiden_max_size(graph, max_cluster_size, seed, weighted) -> dict[str, list[int]]
   返回: {"0": [node1, node2], "1": [node3, node4], ...}
   - 同上，但只包含 is_final_cluster=True 的社区
   
3. compute_leiden_communities(graph, max_cluster_size, seed) -> tuple[dict, list]
   返回: (community_info, community_mapping)
   
   community_info: {
       "0": {
           "level": int,
           "nodes": [node1, node2, ...],
           "is_final_cluster": bool,
           "parent_cluster": str | None
       },
       ...
   }
   
   community_mapping: 原始的partition对象列表（仅供参考）
   
重要注意事项:
- 所有社区ID都必须是字符串类型
- parent_cluster为None时，转为字符串"None"
- 节点的类型要与输入图的节点类型一致
- is_final_cluster 标记叶子社区
    """)
    
    return {
        "c_n_mapping": c_n_mapping,
        "c_n_mapping_hier": c_n_mapping_hier,
        "community_info": community_info,
        "hier_tree": hier_tree,
        "community_mapping_raw": community_mapping_hier
    }


def test_with_sample_graph():
    """创建测试图并运行调试"""
    # 创建一个简单的测试图
    G = nx.karate_club_graph()  # 34个节点的经典测试图
    
    # 添加权重
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.5)
    
    print(f"测试图信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  节点类型示例: {type(list(G.nodes())[0])}")
    print(f"  前5个节点: {list(G.nodes())[:5]}\n")
    
    results = debug_leiden_outputs(G, max_cluster_size=10)
    
    return results


if __name__ == "__main__":
    results = test_with_sample_graph()
    
    # 保存结果供进一步分析
    print("\n" + "=" * 80)
    print(results)


import json

import os

from py2neo import Graph



# --- 配置区 ---

NEO4J_URI = "bolt://localhost:7687"

NEO4J_USER = "neo4j"

# ↓↓↓ 把 "YOUR_NEW_PASSWORD" 替换成您为Neo4j设置的新密码 ↓↓↓

NEO4J_PASSWORD = "123456" 



# --- 特征工程函数 ---

def calculate_structural_features(graph):

    """

    计算并存储结构特征 (入度和出度)

    """

    print("[*] 正在计算所有进程节点的 'in_degree' (入度) 和 'out_degree' (出度)...")

    

    # Cypher 查询：计算每个Process节点的入度和出度

    # [V4 最终修正版]: 使用最兼容的 size( (pattern) ) 语法

    query = """

    MATCH (p:Process)

    WITH p,

         size((p)<-[:CREATED]-()) AS in_degree_created,

         size((p)-[:CREATED]->()) AS out_degree_created,

         size((p)-[:ACCESSED_FILE]->()) AS out_degree_file

    SET p.in_degree = in_degree_created

    SET p.out_degree = out_degree_created + out_degree_file

    RETURN count(p)

    """

    

    try:

        result = graph.run(query).data()

        print(f"[+] 成功为 {result[0]['count(p)']} 个进程节点计算了结构特征。")

    except Exception as e:

        print(f"[!] 错误: 计算结构特征时失败: {e}")

        print("[!] 请检查您的Cypher查询语法和Neo4j版本。")





def calculate_semantic_features(graph):

    """

    计算并存储语义特征 (是否为黑名单程序/访问敏感路径)

    """

    print("[*] 正在计算语义特征 (is_suspicious)...")

    

    # Cypher 查询：根据规则，为可疑进程设置 'is_suspicious' 标记

    query = """

    MATCH (p:Process)

    WHERE 

        // 规则1: 访问了机密文件

        EXISTS((p)-[:ACCESSED_FILE]->(:File {path: '/home/student/secret/secret.txt'})) OR

        // 规则2: 写入了攻击者目录

        EXISTS((p)-[:ACCESSED_FILE]->(:File {path: '/home/attacker/output.txt'})) OR

        // 规则3: 是一个已知的恶意程序名

        p.name IN ['program11', 'program10', 'program9'] OR

        // 规则4: 执行了bash (作为示例规则)

        p.executable ENDS WITH '/bash'

    SET p.is_suspicious = 1

    RETURN count(p)

    """

    

    try:

        result = graph.run(query).data()

        print(f"[+] 成功标记了 {result[0]['count(p)']} 个进程为可疑。")

    except Exception as e:

        print(f"[!] 错误: 计算语义特征时失败: {e}")



    # Cypher 查询：为所有其他进程设置默认值 0

    query_default = """

    MATCH (p:Process)

    WHERE p.is_suspicious IS NULL

    SET p.is_suspicious = 0

    """

    graph.run(query_default)



# --- 主程序区 ---

def main():

    print("[*] 开始连接到Neo4j数据库...")

    try:

        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # 验证连接

        graph.run("RETURN 1")

        print("[+] 成功连接到Neo4j数据库。")

    except Exception as e:

        print(f"[!] 错误: 无法连接到Neo4j数据库: {e}")

        print("[!] 请检查您的NEO4J_PASSWORD是否已正确填写。")

        return

    

    # 1. 计算并存储结构特征

    calculate_structural_features(graph)

    

    # 2. 计算并存储语义特征

    calculate_semantic_features(graph)

    

    print("[*] 特征工程完毕！")



# --- 脚本入口 ---

if __name__ == "__main__":

    main()
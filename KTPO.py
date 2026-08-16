import random
import numpy as np
import pandas as pd
import os
from pathlib import Path


# -------------------------- 1. 数据加载与基础函数 --------------------------
def load_case_data(file_path):
    """
    读取案例数据，动态构建工厂信息和工艺节点映射
    返回：orders列表, factory_cap字典, tech_to_factories字典
    """
    df_orders = pd.read_excel(file_path)
    orders = df_orders.to_dict('records')

    # 从订单数据中提取所有工厂
    all_factories = set()
    for order in orders:
        facs = order['可处理工厂'].split(',')
        all_factories.update(facs)

    # 为每个工厂生成容量（根据工厂数量动态调整）
    factory_count = len(all_factories)
    base_cap = random.Random(42).randint(1500, 2500)
    factory_cap = {f: base_cap + i * 100 for i, f in enumerate(sorted(all_factories, key=lambda x: int(x[1:])))}

    # 动态构建工艺节点到工厂的映射
    tech_to_factories = {}
    for order in orders:
        tech = order['工艺节点']
        facs = order['可处理工厂'].split(',')
        if tech not in tech_to_factories:
            tech_to_factories[tech] = set()
        tech_to_factories[tech].update(facs)

    return orders, factory_cap, tech_to_factories


def decode_and_evaluate(individual, orders, factory_cap):
    """
    解码个体（机器分配串 + 排序串），计算目标值
    输入：
        individual: 列表 [machine_assignment, sequence]
            - machine_assignment: 长度=订单数，元素为工厂名
            - sequence: 长度=订单数，元素为订单索引（全局优先级）
        orders, factory_cap
    输出：makespan, otd (otd越大越好)
    """
    machine_assign = individual[0]
    sequence = individual[1]
    num_orders = len(orders)

    # 按机器分配分组订单索引
    factory_orders = {fac: [] for fac in factory_cap.keys()}
    for order_idx in range(num_orders):
        fac = machine_assign[order_idx]
        # 确保分配的工厂在可处理范围内
        if fac not in orders[order_idx]['可处理工厂'].split(','):
            # 如果非法，强制分配到第一个可处理工厂
            fac = orders[order_idx]['可处理工厂'].split(',')[0]
        factory_orders[fac].append(order_idx)

    # 按全局排序串对每个工厂内的订单排序
    seq_rank = {order_idx: pos for pos, order_idx in enumerate(sequence)}
    for fac in factory_orders:
        factory_orders[fac].sort(key=lambda x: seq_rank[x])

    # 初始化工厂状态
    factory_state = {}
    for fac, cap in factory_cap.items():
        factory_state[fac] = {
            'available_time': 0,
            'remaining_cap': cap,
            'current_batch_end': 0
        }

    order_completion = {}  # 记录每个订单完成时间

    # 依次处理每个工厂的订单序列
    for fac, order_list in factory_orders.items():
        for order_idx in order_list:
            order = orders[order_idx]
            batch_size = order['订单批量（片）']
            proc_time = order['处理时间（周期T）']
            due_time = order['交付时间（周期T）']
            st = factory_state[fac]

            if st['remaining_cap'] >= batch_size:
                # 加入当前批次
                completion_time = st['current_batch_end']
                st['remaining_cap'] -= batch_size
                if proc_time > (st['current_batch_end'] - st['available_time']):
                    st['current_batch_end'] = st['available_time'] + proc_time
                    completion_time = st['current_batch_end']
            else:
                # 开启新批次
                st['available_time'] = st['current_batch_end']
                st['remaining_cap'] = factory_cap[fac] - batch_size
                st['current_batch_end'] = st['available_time'] + proc_time
                completion_time = st['current_batch_end']

            order_completion[order_idx] = completion_time

    # 计算目标
    makespan = max(order_completion.values()) if order_completion else 0
    on_time = sum(1 for i in range(num_orders)
                  if order_completion[i] <= orders[i]['交付时间（周期T）'])
    otd = (on_time / num_orders) * 100 if num_orders > 0 else 0

    return round(makespan, 2), round(otd, 2)


# -------------------------- 2. 知识初始化 --------------------------
def knowledge_initialization(orders, factory_cap, pop_size, seed=None):
    """
    混合启发式初始化：
    - 启发式1：优先分配交付早、批量大的订单到负载轻的工厂（降低拖期）
    - 启发式2：优先分配处理时间短的订单到负载轻的工厂（降低makespan）
    - 随机个体保持多样性
    """
    rng = random.Random(seed)
    num_orders = len(orders)
    factories = list(factory_cap.keys())
    population = []

    # 启发式1：基于拖期
    for _ in range(int(pop_size * 0.3)):
        # 计算初始负载
        fac_load = {f: 0 for f in factories}
        machine_assign = [None] * num_orders
        # 按交付时间升序，批量降序排序
        sorted_indices = sorted(range(num_orders),
                                key=lambda i: (orders[i]['交付时间（周期T）'], -orders[i]['订单批量（片）']))
        for idx in sorted_indices:
            eligible = orders[idx]['可处理工厂'].split(',')
            # 选择负载最小的可处理工厂
            best_fac = min(eligible, key=lambda f: fac_load[f])
            machine_assign[idx] = best_fac
            fac_load[best_fac] += orders[idx]['处理时间（周期T）'] * orders[idx]['订单批量（片）']
        sequence = sorted(range(num_orders), key=lambda i: (orders[i]['交付时间（周期T）'], -orders[i]['订单批量（片）']))
        population.append([machine_assign, sequence])

    # 启发式2：基于makespan
    for _ in range(int(pop_size * 0.3)):
        fac_load = {f: 0 for f in factories}
        machine_assign = [None] * num_orders
        # 按处理时间降序（优先安排长工序）
        sorted_indices = sorted(range(num_orders), key=lambda i: -orders[i]['处理时间（周期T）'])
        for idx in sorted_indices:
            eligible = orders[idx]['可处理工厂'].split(',')
            best_fac = min(eligible, key=lambda f: fac_load[f])
            machine_assign[idx] = best_fac
            fac_load[best_fac] += orders[idx]['处理时间（周期T）'] * orders[idx]['订单批量（片）']
        sequence = sorted(range(num_orders), key=lambda i: -orders[i]['处理时间（周期T）'])
        population.append([machine_assign, sequence])

    # 随机个体
    while len(population) < pop_size:
        machine_assign = []
        for i in range(num_orders):
            eligible = orders[i]['可处理工厂'].split(',')
            machine_assign.append(rng.choice(eligible))
        sequence = rng.sample(range(num_orders), num_orders)
        population.append([machine_assign, sequence])

    return population


# -------------------------- 3. NSGA-II 核心组件 --------------------------
def non_dominated_sorting(objectives):
    """非支配排序，返回fronts列表"""
    N = len(objectives)
    S = [[] for _ in range(N)]
    n = [0] * N
    rank = [0] * N
    fronts = [[]]

    for i in range(N):
        for j in range(N):
            if i == j: continue
            # 目标1：makespan越小越好；目标2：OTD越大越好
            if (objectives[i][0] <= objectives[j][0] and objectives[i][1] >= objectives[j][1]):
                if (objectives[i][0] < objectives[j][0] or objectives[i][1] > objectives[j][1]):
                    S[i].append(j)
            elif (objectives[j][0] <= objectives[i][0] and objectives[j][1] >= objectives[i][1]):
                if (objectives[j][0] < objectives[i][0] or objectives[j][1] > objectives[i][1]):
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    rank[j] = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(objectives, front):
    """计算拥挤距离"""
    dist = [0.0] * len(front)
    if len(front) <= 2:
        return [float('inf')] * len(front)

    for obj_idx in range(2):
        sorted_idx = sorted(range(len(front)), key=lambda x: objectives[front[x]][obj_idx])
        dist[sorted_idx[0]] = float('inf')
        dist[sorted_idx[-1]] = float('inf')
        obj_min = objectives[front[sorted_idx[0]]][obj_idx]
        obj_max = objectives[front[sorted_idx[-1]]][obj_idx]
        if obj_max == obj_min: continue
        for i in range(1, len(front) - 1):
            dist[sorted_idx[i]] += (objectives[front[sorted_idx[i + 1]]][obj_idx] -
                                    objectives[front[sorted_idx[i - 1]]][obj_idx]) / (obj_max - obj_min)
    return dist


def tournament_selection(pop, objectives, fronts, rank_dict, crowd_dict, tour_size=2):
    """二元锦标赛选择"""
    selected = []
    pop_size = len(pop)
    for _ in range(pop_size):
        candidates = random.sample(range(pop_size), tour_size)
        best = min(candidates, key=lambda x: rank_dict[x])
        same_rank = [c for c in candidates if rank_dict[c] == rank_dict[best]]
        if len(same_rank) > 1:
            best = max(same_rank, key=lambda x: crowd_dict[x])
        selected.append([pop[best][0][:], pop[best][1][:]])  # 深拷贝
    return selected


def order_crossover(p1, p2):
    """顺序交叉（OX）用于排序串，同时处理机器分配串的均匀交叉"""
    # 机器分配串均匀交叉
    child_machine = []
    for i in range(len(p1)):
        if random.random() < 0.5:
            child_machine.append(p1[i])
        else:
            child_machine.append(p2[i])

    # 排序串OX交叉
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child_seq = [None] * size
    child_seq[a:b + 1] = p1[a:b + 1]
    ptr = 0
    for gene in p2:
        if gene not in child_seq:
            while child_seq[ptr] is not None:
                ptr += 1
            child_seq[ptr] = gene
    return [child_machine, child_seq]


def swap_mutation(individual, orders, mut_prob=0.2):
    """变异：机器分配串位变异（在可处理工厂内随机选择），排序串交换变异"""
    machine = individual[0][:]
    seq = individual[1][:]
    num_orders = len(orders)

    # 机器分配串变异
    for i in range(num_orders):
        if random.random() < mut_prob / num_orders:
            eligible = orders[i]['可处理工厂'].split(',')
            machine[i] = random.choice(eligible)

    # 排序串交换变异
    if random.random() < mut_prob:
        i, j = random.sample(range(num_orders), 2)
        seq[i], seq[j] = seq[j], seq[i]

    return [machine, seq]


# -------------------------- 4. 差分进化模块（离散适配） --------------------------
def de_mutation(pop, F=0.5):
    """DE/rand/1 变异，对机器分配和排序分别处理"""
    num_orders = len(pop[0][0])
    new_pop = []
    for idx in range(len(pop)):
        candidates = random.sample([i for i in range(len(pop)) if i != idx], 3)
        base, r1, r2 = pop[candidates[0]], pop[candidates[1]], pop[candidates[2]]
        # 机器分配：随机继承或投票
        child_machine = []
        for j in range(num_orders):
            if random.random() < F:
                # 选择三个个体中多数派或随机
                choices = [base[0][j], r1[0][j], r2[0][j]]
                child_machine.append(random.choice(choices))
            else:
                child_machine.append(pop[idx][0][j])
        # 排序串：基于顺序的DE变异（类似OX，取三个个体部分片段）
        child_seq = pop[idx][1][:]
        if random.random() < F:
            # 使用部分映射组合
            a, b = sorted(random.sample(range(num_orders), 2))
            segment = base[1][a:b + 1]
            # 从r1中填充缺失基因
            temp = [x for x in r1[1] if x not in segment]
            child_seq = temp[:a] + segment + temp[a:]
            if len(child_seq) != num_orders:
                # 回退为原排序
                child_seq = pop[idx][1][:]
        new_pop.append([child_machine, child_seq])
    return new_pop


def de_crossover(target, mutant, CR=0.8):
    """二项式交叉"""
    num_orders = len(target[0])
    child_machine = []
    for j in range(num_orders):
        if random.random() < CR:
            child_machine.append(mutant[0][j])
        else:
            child_machine.append(target[0][j])
    # 排序串使用OX交叉
    child_seq = order_crossover(target[1], mutant[1])[1]
    return [child_machine, child_seq]


# -------------------------- 5. 知识局部搜索 --------------------------
def local_search_reduce_tardiness(individual, orders, factory_cap):
    """
    局部搜索1：降低总拖期
    对每个工厂，尝试交换相邻订单，若交换后该工厂内订单总拖期下降且makespan不增加，则接受
    """
    machine = individual[0][:]
    seq = individual[1][:]
    improved = True
    num_orders = len(orders)

    # 构建每个工厂的订单列表
    factory_orders = {fac: [] for fac in factory_cap}
    for i in range(num_orders):
        fac = machine[i]
        if fac not in orders[i]['可处理工厂'].split(','):
            fac = orders[i]['可处理工厂'].split(',')[0]
        factory_orders[fac].append(i)
    seq_rank = {idx: pos for pos, idx in enumerate(seq)}
    for fac in factory_orders:
        factory_orders[fac].sort(key=lambda x: seq_rank[x])

    # 对每个工厂进行局部搜索
    for fac in factory_orders:
        lst = factory_orders[fac]
        if len(lst) < 2:
            continue

        # 计算当前工厂拖期
        def compute_tardiness(order_list):
            # 简化：只计算该工厂内订单的拖期，需要模拟该工厂的加工
            st = {'available_time': 0, 'remaining_cap': factory_cap[fac], 'current_batch_end': 0}
            total_tardy = 0
            max_completion = 0
            for oi in order_list:
                order = orders[oi]
                batch = order['订单批量（片）']
                proc = order['处理时间（周期T）']
                due = order['交付时间（周期T）']
                if st['remaining_cap'] >= batch:
                    comp = st['current_batch_end']
                    st['remaining_cap'] -= batch
                    if proc > (st['current_batch_end'] - st['available_time']):
                        st['current_batch_end'] = st['available_time'] + proc
                        comp = st['current_batch_end']
                else:
                    st['available_time'] = st['current_batch_end']
                    st['remaining_cap'] = factory_cap[fac] - batch
                    st['current_batch_end'] = st['available_time'] + proc
                    comp = st['current_batch_end']
                if comp > due:
                    total_tardy += (comp - due)
                max_completion = max(max_completion, comp)
            return total_tardy, max_completion

        original_tardy, original_makespan_fac = compute_tardiness(lst)
        # 尝试交换相邻订单
        for i in range(len(lst) - 1):
            new_lst = lst[:]
            new_lst[i], new_lst[i + 1] = new_lst[i + 1], new_lst[i]
            new_tardy, new_makespan_fac = compute_tardiness(new_lst)
            if new_tardy < original_tardy and new_makespan_fac <= original_makespan_fac + 1e-6:
                # 接受交换，更新列表
                lst = new_lst
                original_tardy = new_tardy
                original_makespan_fac = new_makespan_fac
                # 更新全局序列中的相对顺序
                # 简单的做法：重新构建seq中该工厂部分的顺序
                # 这里为了简洁，仅更新本地的lst，后续重新生成全局seq
        # 将优化后的顺序写回全局seq
        # 找到该工厂订单在seq中的位置，按lst顺序重新排列
        pos_in_seq = [seq_rank[oi] for oi in lst]
        sorted_pos = sorted(pos_in_seq)
        for new_pos, oi in zip(sorted_pos, lst):
            seq[new_pos] = oi

    return [machine, seq]


def local_search_reduce_makespan(individual, orders, factory_cap):
    """
    局部搜索2：降低makespan
    尝试将某些订单移动到其他可处理工厂，若makespan下降且OTD不下降，则接受
    """
    machine = individual[0][:]
    seq = individual[1][:]
    num_orders = len(orders)

    # 简单实现：随机尝试移动若干订单
    for _ in range(min(10, num_orders)):
        i = random.randrange(num_orders)
        eligible = orders[i]['可处理工厂'].split(',')
        if len(eligible) < 2:
            continue
        current_fac = machine[i]
        other_facs = [f for f in eligible if f != current_fac]
        if not other_facs:
            continue
        new_fac = random.choice(other_facs)
        # 复制个体测试
        test_ind = [machine[:], seq[:]]
        test_ind[0][i] = new_fac
        # 重新计算目标
        mk_new, otd_new = decode_and_evaluate(test_ind, orders, factory_cap)
        mk_old, otd_old = decode_and_evaluate([machine, seq], orders, factory_cap)
        if mk_new < mk_old and otd_new >= otd_old - 1e-6:
            machine[i] = new_fac
    return [machine, seq]


# -------------------------- 6. KTPO主算法 --------------------------
def solve_case_ktpo(orders, factory_cap, pop_size=60, max_gen=50):
    """
    KTPO主流程：双种群（NSGA-II + DE）并行，知识初始化，局部搜索
    """
    num_orders = len(orders)
    # 参数调整（大规模案例降低计算量）
    if num_orders > 400:
        pop_size = min(pop_size, 40)
        max_gen = min(max_gen, 30)
    elif num_orders > 150:
        pop_size = min(pop_size, 50)
        max_gen = min(max_gen, 40)

    # 初始化总种群
    pop = knowledge_initialization(orders, factory_cap, pop_size, seed=42)

    # 外部帕累托档案
    archive = []

    for gen in range(max_gen):
        # 评估当前种群
        objectives = []
        for ind in pop:
            mk, otd = decode_and_evaluate(ind, orders, factory_cap)
            objectives.append([mk, otd])
        objectives = np.array(objectives)

        # 非支配排序和拥挤度
        fronts = non_dominated_sorting(objectives)
        rank_dict = {}
        crowd_dict = {}
        for f_idx, front in enumerate(fronts):
            dists = crowding_distance(objectives, front)
            for i, idx in enumerate(front):
                rank_dict[idx] = f_idx
                crowd_dict[idx] = dists[i]

        # 种群划分：前一半用NSGA-II，后一半用DE
        half = pop_size // 2
        nsga_pop = pop[:half]
        de_pop = pop[half:]

        # ---- NSGA-II子代生成 ----
        mating_pool = tournament_selection(nsga_pop, objectives[:half], fronts,
                                           {i: rank_dict[i] for i in range(half)},
                                           {i: crowd_dict[i] for i in range(half)})
        nsga_offspring = []
        for i in range(0, len(mating_pool), 2):
            p1 = mating_pool[i]
            p2 = mating_pool[i + 1] if i + 1 < len(mating_pool) else mating_pool[i]
            if random.random() < 0.8:
                c1 = order_crossover(p1[1], p2[1])
                c1 = [c1[0], c1[1]]
                c2 = order_crossover(p2[1], p1[1])
                c2 = [c2[0], c2[1]]
                # 机器分配串交叉
                c1[0] = p1[0][:] if random.random() < 0.5 else p2[0][:]
                c2[0] = p1[0][:] if random.random() < 0.5 else p2[0][:]
            else:
                c1 = [p1[0][:], p1[1][:]]
                c2 = [p2[0][:], p2[1][:]]
            nsga_offspring.append(swap_mutation(c1, orders, mut_prob=0.2))
            nsga_offspring.append(swap_mutation(c2, orders, mut_prob=0.2))
        nsga_offspring = nsga_offspring[:half]

        # ---- DE子代生成 ----
        de_mutants = de_mutation(de_pop, F=0.5)
        de_offspring = []
        for i in range(len(de_pop)):
            child = de_crossover(de_pop[i], de_mutants[i], CR=0.8)
            # 修复非法解
            for j in range(num_orders):
                if child[0][j] not in orders[j]['可处理工厂'].split(','):
                    child[0][j] = orders[j]['可处理工厂'].split(',')[0]
            de_offspring.append(child)

        # 合并子代
        combined_offspring = nsga_offspring + de_offspring
        combined_offspring = combined_offspring[:pop_size]

        # 评估子代
        off_objectives = []
        for ind in combined_offspring:
            mk, otd = decode_and_evaluate(ind, orders, factory_cap)
            off_objectives.append([mk, otd])
        off_objectives = np.array(off_objectives)

        # 合并父子种群，筛选下一代
        all_pop = pop + combined_offspring
        all_obj = np.vstack([objectives, off_objectives])

        all_fronts = non_dominated_sorting(all_obj)
        new_pop = []
        new_obj = []
        for front in all_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(all_pop[idx])
                    new_obj.append(all_obj[idx])
            else:
                # 拥挤距离选择
                dists = crowding_distance(all_obj, front)
                sorted_idx = sorted(front, key=lambda x: dists[front.index(x)], reverse=True)
                for idx in sorted_idx[:pop_size - len(new_pop)]:
                    new_pop.append(all_pop[idx])
                    new_obj.append(all_obj[idx])
                break
        pop = new_pop

        # ---- 知识局部搜索 ----
        for i in range(len(pop)):
            pop[i] = local_search_reduce_tardiness(pop[i], orders, factory_cap)
            pop[i] = local_search_reduce_makespan(pop[i], orders, factory_cap)

        # 更新外部档案
        for ind in pop:
            mk, otd = decode_and_evaluate(ind, orders, factory_cap)
            archive.append([ind, mk, otd])

        # 去除档案中被支配的
        if archive:
            arch_obj = np.array([[a[1], a[2]] for a in archive])
            keep = []
            for i in range(len(archive)):
                dominated = False
                for j in range(len(archive)):
                    if i == j: continue
                    if (arch_obj[j][0] <= arch_obj[i][0] and arch_obj[j][1] >= arch_obj[i][1]) and \
                            (arch_obj[j][0] < arch_obj[i][0] or arch_obj[j][1] > arch_obj[i][1]):
                        dominated = True
                        break
                if not dominated:
                    keep.append(archive[i])
            archive = keep

        if (gen + 1) % 10 == 0:
            print(f"  进度: 第 {gen + 1}/{max_gen} 代, 档案大小: {len(archive)}")

    # 最终提取帕累托最优
    final_pop = pop + [a[0] for a in archive]
    final_obj = []
    for ind in final_pop:
        mk, otd = decode_and_evaluate(ind, orders, factory_cap)
        final_obj.append([mk, otd])
    final_obj = np.array(final_obj)

    final_fronts = non_dominated_sorting(final_obj)
    pareto_indices = final_fronts[0]
    pareto_obj = final_obj[pareto_indices]
    pareto_individuals = [final_pop[i] for i in pareto_indices]

    # 去重
    unique_pareto = []
    unique_obj = []
    for ind, obj in zip(pareto_individuals, pareto_obj):
        if obj.tolist() not in unique_obj:
            unique_obj.append(obj.tolist())
            unique_pareto.append(ind)

    return np.array(unique_obj), unique_pareto


# -------------------------- 7. 批量求解与保存 --------------------------
if __name__ == "__main__":
    input_dir = 'wafer_scheduling_cases_new'
    output_dir = 'ktpo_results'
    os.makedirs(output_dir, exist_ok=True)

    scale_configs = [
        {'label': '10f_100o', 'pop_size': 60, 'max_gen': 50},
        {'label': '20f_200o', 'pop_size': 50, 'max_gen': 40},
        {'label': '30f_500o', 'pop_size': 40, 'max_gen': 30},
    ]

    summary_data = []

    for config in scale_configs:
        label = config['label']
        scale_input_dir = os.path.join(input_dir, label)

        if not os.path.exists(scale_input_dir):
            print(f"跳过 {label}（目录不存在）")
            continue

        scale_output_dir = os.path.join(output_dir, label)
        os.makedirs(scale_output_dir, exist_ok=True)

        case_files = sorted([f for f in os.listdir(scale_input_dir) if f.endswith('.xlsx')])

        print(f"\n{'=' * 60}")
        print(f"开始求解 {label} 规模的案例（共 {len(case_files)} 个）")
        print(f"{'=' * 60}")

        for case_file in case_files:
            case_name = case_file.replace('.xlsx', '')
            file_path = os.path.join(scale_input_dir, case_file)

            print(f"\n正在求解案例: {case_name} ...")

            orders, factory_cap, _ = load_case_data(file_path)
            print(f"  订单数量: {len(orders)}, 工厂数量: {len(factory_cap)}")

            # 运行KTPO
            pareto_obj, pareto_ind = solve_case_ktpo(
                orders, factory_cap,
                pop_size=config['pop_size'],
                max_gen=config['max_gen']
            )

            # 保存帕累托解
            df_pareto = pd.DataFrame(pareto_obj, columns=['最大完工时间(T)', '准时交付率(%)'])
            df_pareto.insert(0, 'Pareto解编号', range(1, len(df_pareto) + 1))

            output_path = os.path.join(scale_output_dir, f'{case_name}_ktpo_pareto.xlsx')
            df_pareto.to_excel(output_path, index=False)

            best_otd_idx = np.argmax(pareto_obj[:, 1])
            best_mk_idx = np.argmin(pareto_obj[:, 0])
            summary_data.append({
                '规模类别': label,
                '案例编号': case_name,
                '订单数量': len(orders),
                '工厂数量': len(factory_cap),
                'Pareto解数量': len(pareto_obj),
                '最优OTD(%)': pareto_obj[best_otd_idx, 1],
                '对应Makespan(T)': pareto_obj[best_otd_idx, 0],
                '最优Makespan(T)': pareto_obj[best_mk_idx, 0],
                '对应OTD(%)': pareto_obj[best_mk_idx, 1]
            })
            print(
                f"  Pareto解数量: {len(pareto_obj)}, 最优OTD: {pareto_obj[best_otd_idx, 1]}%, 最优Makespan: {pareto_obj[best_mk_idx, 0]}T")

    # 保存汇总
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, '00_所有案例结果汇总_ktpo.xlsx')
    df_summary.to_excel(summary_path, index=False)
    print(f"\n全部完成！结果保存于 {output_dir}")
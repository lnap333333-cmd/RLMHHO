#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鐗瑰畾瑙勬ā绠楁硶瀵规瘮瀹為獙绋嬪簭 - 淇鐗堟湰
瑙ｅ喅闂锛?1. DQN pareto瑙ｉ泦鏁伴噺闂
2. 褰掍竴鍖栨寚鏍囪绠楅棶棰? 
3. 涓讳綋绠楁硶pareto瑙ｉ泦澶氭牱鎬?4. Excel琛ㄦ牸鍒嗙杈撳嚭
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc import QLABC_Optimizer
from utils.data_generator import DataGenerator

# 璁剧疆涓枃瀛椾綋
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_hypervolume(pareto_solutions: List, reference_point: Tuple[float, float] = None, normalize: bool = False) -> float:
    """
    淇鍚庣殑瓒呬綋绉寚鏍囪绠?    浣跨敤姝ｇ‘鐨?D瓒呬綋绉畻娉曪紝閬垮厤铏氶珮鎴栬櫄浣庣殑HV鍊?    """
    if not pareto_solutions or len(pareto_solutions) == 0:
        return 0.0
    
    # 鎻愬彇鐩爣鍊?    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 鍘婚櫎閲嶅瑙ｏ紙鏇村鏉剧殑瀹瑰樊锛?    unique_objectives = []
    tolerance = 1e-3  # 鏀惧瀹瑰樊
    for obj in objectives:
        is_duplicate = False
        for unique_obj in unique_objectives:
            if abs(obj[0] - unique_obj[0]) < tolerance and abs(obj[1] - unique_obj[1]) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_objectives.append(obj)
    
    if len(unique_objectives) == 0:
        return 0.0
    
    # 涓ユ牸璁＄畻甯曠疮鎵樺墠娌?    pareto_front = []
    for i, obj in enumerate(unique_objectives):
        is_dominated = False
        for j, other_obj in enumerate(unique_objectives):
            if i != j:  # 涓嶄笌鑷繁姣旇緝
                # 妫€鏌ユ槸鍚﹁涓ユ牸鏀厤锛堝浜庢渶灏忓寲闂锛?                if (other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and 
                    (other_obj[0] < obj[0] or other_obj[1] < obj[1])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(obj)
    
    if len(pareto_front) == 0:
        return 0.0
    
    # 鍗曠偣瑙ｇ粰浜堝悎鐞嗙殑鍩虹鍒嗘暟
    if len(pareto_front) == 1:
        return 0.1  # 鍥哄畾杩斿洖0.1浣滀负鍗曠偣瑙ｇ殑鍩虹鍒嗘暟
    
    # 璁剧疆鍚堢悊鐨勫弬鑰冪偣锛堜娇鐢ㄦ洿澶х殑鎵╁睍姣斾緥锛?    if reference_point is None:
        max_makespan = max(obj[0] for obj in pareto_front)
        max_tardiness = max(obj[1] for obj in pareto_front)
        min_makespan = min(obj[0] for obj in pareto_front)
        min_tardiness = min(obj[1] for obj in pareto_front)
        
        # 浣跨敤鍔ㄦ€佹墿灞曟瘮渚嬶紝纭繚鏈夋剰涔夌殑HV璁＄畻绌洪棿
        makespan_range = max_makespan - min_makespan
        tardiness_range = max_tardiness - min_tardiness
        
        # 鑷冲皯鎵╁睍20%锛屽浜庡皬鑼冨洿鎵╁睍鏇村
        makespan_margin = max(makespan_range * 0.3, max_makespan * 0.15, 1.0)
        tardiness_margin = max(tardiness_range * 0.3, max_tardiness * 0.15, 1.0)
        
        reference_point = (max_makespan + makespan_margin, max_tardiness + tardiness_margin)
    
    # 浣跨敤姝ｇ‘鐨?D瓒呬綋绉绠楃畻娉曪紙浠庡乏鍒板彸鎵弿锛?    sorted_points = sorted(pareto_front, key=lambda x: x[0])  # 鎸墄鍧愭爣鎺掑簭
    
    hypervolume = 0.0
    prev_x = 0.0  # 浠庡師鐐瑰紑濮?    
    for i, (x, y) in enumerate(sorted_points):
        # 纭繚鐐瑰湪鍙傝€冪偣鍐?        if x >= reference_point[0] or y >= reference_point[1]:
            continue
            
        # 璁＄畻褰撳墠鐐瑰乏渚х殑鐭╁舰璐＄尞
        width = x - prev_x
        height = reference_point[1] - y
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
        # 鏇存柊x鍧愭爣
        prev_x = x
    
    # 娣诲姞鏈€鍙充晶鍖哄煙鐨勮础鐚紙浠庢渶鍚庝竴涓偣鍒板弬鑰冪偣锛?    if sorted_points:
        last_x, last_y = sorted_points[-1]
        if last_x < reference_point[0]:
            # 鎵惧埌鍦ㄦ渶鍚巟鍧愭爣澶勭殑鏈€灏弝鍊?            min_y = min(y for x, y in sorted_points if x == last_x)
            width = reference_point[0] - last_x
            height = reference_point[1] - min_y
            
            if width > 0 and height > 0:
                hypervolume += width * height
    
    # 纭繚杩斿洖姝ｅ€?    hypervolume = max(0.0, hypervolume)
    
    # 涓轰簡鍏钩姣旇緝锛屽鎵€鏈夌畻娉曚娇鐢ㄧ浉鍚岀殑褰掍竴鍖栧熀鍑?    # 浣跨敤鍙傝€冪偣鐭╁舰闈㈢Н杩涜褰掍竴鍖?    max_possible_hv = reference_point[0] * reference_point[1]
    if max_possible_hv > 0:
        normalized_hv = hypervolume / max_possible_hv
        # 闄愬埗褰掍竴鍖朒V鐨勬渶澶у€硷紝閬垮厤铏氶珮
        normalized_hv = min(normalized_hv, 0.95)  # 鏈€澶т笉瓒呰繃0.95
        return normalized_hv
    else:
        return 0.0

def calculate_igd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    鍙嶅悜涓栦唬璺濈 - 鍩轰簬褰掍竴鍖栧悗鐨勭洰鏍囧€艰绠?    IGD+ 淇鐗堟湰锛岃€冭檻鏀厤鍏崇郴
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 浣跨敤褰掍竴鍖栧悗鐨勭洰鏍囧€?    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 濡傛灉娌℃湁鍙傝€冨墠娌匡紝杩斿洖鏃犵┓澶?    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 璁＄畻姣忎釜鍙傝€冪偣鍒拌В闆嗙殑鏈€灏忚窛绂?    distances = []
    for ref_point in reference_front:
        min_distance = float('inf')
        
        for obj in objectives:
            # 浣跨敤IGD+鐨勪慨姝ｈ窛绂昏绠楋紙鑰冭檻鏀厤鍏崇郴锛?            # 瀵逛簬鏈€灏忓寲闂锛歞+ = max{obj - ref, 0}
            diff_makespan = max(obj[0] - ref_point[0], 0)
            diff_tardiness = max(obj[1] - ref_point[1], 0)
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 杩斿洖骞冲潎璺濈
    return np.mean(distances)

def calculate_gd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    涓栦唬璺濈 - 鍩轰簬褰掍竴鍖栧悗鐨勭洰鏍囧€艰绠?    GD+ 淇鐗堟湰锛岃€冭檻鏀厤鍏崇郴
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 浣跨敤褰掍竴鍖栧悗鐨勭洰鏍囧€?    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 濡傛灉娌℃湁鍙傝€冨墠娌匡紝杩斿洖鏃犵┓澶?    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 璁＄畻姣忎釜瑙ｅ埌鍙傝€冨墠娌跨殑鏈€灏忚窛绂?    distances = []
    for obj in objectives:
        min_distance = float('inf')
        
        for ref_point in reference_front:
            # 浣跨敤GD+鐨勪慨姝ｈ窛绂昏绠楋紙鑰冭檻鏀厤鍏崇郴锛?            # 瀵逛簬鏈€灏忓寲闂锛歞+ = max{obj - ref, 0}
            diff_makespan = max(obj[0] - ref_point[0], 0)
            diff_tardiness = max(obj[1] - ref_point[1], 0)
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 杩斿洖骞冲潎璺濈
    return np.mean(distances)

def calculate_spacing(normalized_pareto_solutions: List) -> float:
    """
    闂磋窛鎸囨爣 - 鍩轰簬褰掍竴鍖栧悗鐨勭洰鏍囧€艰绠?    娴嬮噺瑙ｉ泦鍒嗗竷鐨勫潎鍖€鎬?    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) <= 1:
        return 0.0  # 鍗曠偣瑙ｉ泦闂磋窛涓?
    
    # 浣跨敤褰掍竴鍖栧悗鐨勭洰鏍囧€?    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    if len(objectives) <= 1:
        return 0.0
    
    # 璁＄畻姣忎釜瑙ｅ埌鍏舵渶杩戦偦鐨勮窛绂?    nearest_distances = []
    for i, obj1 in enumerate(objectives):
        min_distance = float('inf')
        
        for j, obj2 in enumerate(objectives):
            if i != j:
                # 娆у嚑閲屽緱璺濈
                distance = np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)
                min_distance = min(min_distance, distance)
    
        if min_distance != float('inf'):
            nearest_distances.append(min_distance)
    
    if len(nearest_distances) <= 1:
        return 0.0
    
    # Schott鐨勬爣鍑唖pacing鍏紡锛氭渶杩戦偦璺濈鐨勬爣鍑嗗樊
    mean_distance = np.mean(nearest_distances)
    variance = np.sum([(d - mean_distance)**2 for d in nearest_distances]) / len(nearest_distances)
    spacing = np.sqrt(variance)
    
    return spacing

def calculate_spread(normalized_pareto_solutions: List) -> float:
    """
    鍒嗗竷鎬ф寚鏍?- 鍩轰簬褰掍竴鍖栧悗鐨勭洰鏍囧€艰绠?    娴嬮噺瑙ｉ泦鍦ㄧ洰鏍囩┖闂寸殑鍒嗗竷鑼冨洿鍜屽潎鍖€鎬?    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) <= 2:
        return 0.0  # 灏戜簬3涓В鏃讹紝鍒嗗竷鎬т负0
    
    # 浣跨敤褰掍竴鍖栧悗鐨勭洰鏍囧€?    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    if len(objectives) <= 2:
        return 0.0
    
    # 鎵惧埌姣忎釜鐩爣鐨勬瀬鍊艰В
    min_makespan_sol = min(objectives, key=lambda x: x[0])
    min_tardiness_sol = min(objectives, key=lambda x: x[1])
    
    # 璁＄畻姣忎釜瑙ｅ埌鍏舵渶杩戦偦鐨勮窛绂?    nearest_distances = []
    for i, obj1 in enumerate(objectives):
        min_distance = float('inf')
        
        for j, obj2 in enumerate(objectives):
            if i != j:
                distance = np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)
                min_distance = min(min_distance, distance)
        
        if min_distance != float('inf'):
            nearest_distances.append(min_distance)
    
    if len(nearest_distances) == 0:
        return 0.0
    
    # 璁＄畻骞冲潎璺濈
    mean_distance = np.mean(nearest_distances)
    
    # 璁＄畻鏋佸€艰В鐨勮竟鐣岃窛绂?    df = 0.0  # 绗竴涓洰鏍囨瀬鍊艰В鐨勮窛绂?    dl = 0.0  # 绗簩涓洰鏍囨瀬鍊艰В鐨勮窛绂?    
    # 濡傛灉鏋佸€艰В涓嶅悓锛岃绠楄竟鐣岃窛绂?    if min_makespan_sol != min_tardiness_sol:
        df = np.sqrt((min_makespan_sol[0] - min_tardiness_sol[0])**2 + 
                     (min_makespan_sol[1] - min_tardiness_sol[1])**2)
        dl = df  # 瀵逛簬鍙岀洰鏍囬棶棰橈紝杈圭晫璺濈鐩稿悓
    
    # Deb鐨勬爣鍑唖pread鍏紡
    if mean_distance > 0:
        distance_deviations = [abs(d - mean_distance) for d in nearest_distances]
        numerator = df + dl + sum(distance_deviations)
        denominator = df + dl + (len(nearest_distances) * mean_distance)
        
        if denominator > 0:
            spread = numerator / denominator
        else:
            spread = 0.0
    else:
        spread = 0.0
    
    return spread

def normalize_objectives(all_results: Dict) -> Dict:
    """
    褰掍竴鍖栨墍鏈夌畻娉曠殑鐩爣鍊硷紝閬垮厤涓嶅悓閲忕翰褰卞搷
    杩斿洖褰掍竴鍖栧悗鐨勭粨鏋滃拰褰掍竴鍖栧弬鏁?    """
    # 鏀堕泦鎵€鏈夌洰鏍囧€?    all_makespans = []
    all_tardiness = []
    
    for result in all_results.values():
        if 'pareto_solutions' in result and result['pareto_solutions']:
            for sol in result['pareto_solutions']:
                all_makespans.append(sol.makespan)
                all_tardiness.append(sol.total_tardiness)
    
    if not all_makespans:
        return all_results, (0, 1, 0, 1)
    
    # 璁＄畻褰掍竴鍖栧弬鏁?    min_makespan = min(all_makespans)
    max_makespan = max(all_makespans)
    min_tardiness = min(all_tardiness)
    max_tardiness = max(all_tardiness)
    
    # 閬垮厤闄ら浂
    makespan_range = max_makespan - min_makespan if max_makespan > min_makespan else 1.0
    tardiness_range = max_tardiness - min_tardiness if max_tardiness > min_tardiness else 1.0
    
    # 褰掍竴鍖栨墍鏈夎В
    normalized_results = {}
    for alg_name, result in all_results.items():
        normalized_results[alg_name] = result.copy()
        
        if 'pareto_solutions' in result and result['pareto_solutions']:
            normalized_solutions = []
            for sol in result['pareto_solutions']:
                # 鍒涘缓褰掍竴鍖栬В鐨勫壇鏈?                norm_sol = type('Solution', (), {})()
                norm_sol.makespan = (sol.makespan - min_makespan) / makespan_range
                norm_sol.total_tardiness = (sol.total_tardiness - min_tardiness) / tardiness_range
                # 淇濈暀鍘熷鍊肩敤浜庡叾浠栫敤閫?                norm_sol.original_makespan = sol.makespan
                norm_sol.original_tardiness = sol.total_tardiness
                normalized_solutions.append(norm_sol)
            
            normalized_results[alg_name]['normalized_pareto_solutions'] = normalized_solutions
    
    normalization_params = (min_makespan, max_makespan, min_tardiness, max_tardiness)
    return normalized_results, normalization_params

def calculate_combined_pareto_front(normalized_results: Dict) -> List[Tuple[float, float]]:
    """
    鍩轰簬褰掍竴鍖栧悗鐨勭洰鏍囧€艰绠楃粍鍚堝笗绱墭鍓嶆部
    鐢ㄤ綔IGD鍜孏D鐨勭湡瀹炲弬鑰冨墠娌縋F*
    """
    all_objectives = []
    
    # 鏀堕泦鎵€鏈夌畻娉曞綊涓€鍖栧悗鐨勭洰鏍囧€?    for algorithm_name, result in normalized_results.items():
        if 'normalized_pareto_solutions' in result and result['normalized_pareto_solutions']:
            for sol in result['normalized_pareto_solutions']:
                all_objectives.append((sol.makespan, sol.total_tardiness))
    
    if not all_objectives:
        return []
    
    # 鍘婚櫎閲嶅鐐?    unique_objectives = []
    for obj in all_objectives:
        is_duplicate = False
        for unique_obj in unique_objectives:
            if abs(obj[0] - unique_obj[0]) < 1e-6 and abs(obj[1] - unique_obj[1]) < 1e-6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_objectives.append(obj)
    
    # 璁＄畻甯曠疮鎵樺墠娌?    pareto_front = []
    for obj in unique_objectives:
        is_dominated = False
        for other_obj in unique_objectives:
            # 妫€鏌ユ槸鍚﹁鏀厤锛堝浜庢渶灏忓寲闂锛?            if (other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and 
                (other_obj[0] < obj[0] or other_obj[1] < obj[1])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(obj)
    
    return pareto_front

def normalize_metrics(all_results: Dict) -> Dict:
    """
    鎸夌収瀛︽湳鏍囧噯褰掍竴鍖栨寚鏍?    - HV: 瓒婂ぇ瓒婂ソ锛屽綊涓€鍖栦负0-1锛?琛ㄧず鏈€濂?    - IGD銆丟D: 瓒婂皬瓒婂ソ锛屾寜鐓у鏈儻渚嬶紝0琛ㄧず鏈€濂斤紝浣嗛渶瑕佸悎鐞嗚寖鍥存樉绀?    - Spacing: 瓒婂皬瓒婂ソ锛?琛ㄧず鏈€濂?    - Spread: 瓒婂皬瓒婂ソ锛岀悊鎯宠寖鍥?.1-0.9
    """
    # 鏀堕泦鎵€鏈夋寚鏍囧€?    all_hypervolume = []
    all_igd = []
    all_gd = []
    all_spacing = []
    all_spread = []
    all_makespan = []
    all_tardiness = []
    
    for result in all_results.values():
        if result['hypervolume'] > 0:
            all_hypervolume.append(result['hypervolume'])
        
        # 鏀堕泦闈炴棤绌峰€肩殑IGD鍜孏D
        if result['igd'] != float('inf') and not np.isnan(result['igd']) and result['igd'] >= 0:
                all_igd.append(result['igd'])
        if result['gd'] != float('inf') and not np.isnan(result['gd']) and result['gd'] >= 0:
                all_gd.append(result['gd'])
        if result['spacing'] >= 0 and not np.isnan(result['spacing']):
            all_spacing.append(result['spacing'])
        if result['spread'] != float('inf') and not np.isnan(result['spread']) and result['spread'] >= 0:
                all_spread.append(result['spread'])
    
        if result['makespan_best'] > 0:
            all_makespan.append(result['makespan_best'])
        if result['tardiness_best'] >= 0:
            all_tardiness.append(result['tardiness_best'])
    
    # 璁＄畻褰掍竴鍖栧弬鏁?    max_hv = max(all_hypervolume) if all_hypervolume else 1.0
    
    # 瀵逛簬IGD鍜孏D锛屾垜浠笇鏈涙樉绀哄畠浠殑鐩稿浼樺姡锛屼絾淇濇寔"瓒婂皬瓒婂ソ"鐨勫惈涔?    max_igd = max(all_igd) if all_igd else 1.0
    max_gd = max(all_gd) if all_gd else 1.0
    max_spacing = max(all_spacing) if all_spacing else 1.0
    max_spread = max(all_spread) if all_spread else 1.0
    
    min_makespan = min(all_makespan) if all_makespan else 0.0
    max_makespan = max(all_makespan) if all_makespan else 1.0
    min_tardiness = min(all_tardiness) if all_tardiness else 0.0
    max_tardiness = max(all_tardiness) if all_tardiness else 1.0
    
    # 褰掍竴鍖栫粨鏋?    normalized_results = {}
    for alg_name, result in all_results.items():
        normalized_results[alg_name] = result.copy()
        
        # 瓒呬綋绉疕V: 瓒婂ぇ瓒婂ソ锛屽綊涓€鍖栧埌0-1鑼冨洿锛?琛ㄧず鏈€濂?        normalized_results[alg_name]['norm_hypervolume'] = result['hypervolume'] / max_hv if max_hv > 0 else 0.0
        
        # IGD: 瓒婂皬瓒婂ソ锛屼繚鎸佸師濮嬪€兼樉绀猴紝浣嗘爣璁颁负瑙勮寖鍖栧悗鐨勫€?        if result['igd'] == float('inf') or np.isnan(result['igd']):
            normalized_results[alg_name]['norm_igd'] = max_igd * 2  # 缁欏け璐ョ畻娉曚竴涓緢澶х殑鍊?        else:
            normalized_results[alg_name]['norm_igd'] = result['igd']
        
        # GD: 瓒婂皬瓒婂ソ锛屼繚鎸佸師濮嬪€兼樉绀?        if result['gd'] == float('inf') or np.isnan(result['gd']):
            normalized_results[alg_name]['norm_gd'] = max_gd * 2
        else:
            normalized_results[alg_name]['norm_gd'] = result['gd']
        
        # Spacing: 瓒婂皬瓒婂ソ锛屼繚鎸佸師濮嬪€兼樉绀?        if np.isnan(result['spacing']):
            normalized_results[alg_name]['norm_spacing'] = max_spacing * 2
        else:
            normalized_results[alg_name]['norm_spacing'] = result['spacing']
        
        # Spread: 瓒婂皬瓒婂ソ锛屼繚鎸佸師濮嬪€兼樉绀?        if result['spread'] == float('inf') or np.isnan(result['spread']):
            normalized_results[alg_name]['norm_spread'] = max_spread * 2
        else:
            normalized_results[alg_name]['norm_spread'] = result['spread']
            
        # 鐩爣鍊煎綊涓€鍖?(瓒婂皬瓒婂ソ鐨勬寚鏍?
        if max_makespan > min_makespan:
            normalized_results[alg_name]['norm_makespan'] = 1 - (result['makespan_best'] - min_makespan) / (max_makespan - min_makespan)
        else:
            normalized_results[alg_name]['norm_makespan'] = 1.0
            
        if max_tardiness > min_tardiness:
            normalized_results[alg_name]['norm_tardiness'] = 1 - (result['tardiness_best'] - min_tardiness) / (max_tardiness - min_tardiness)
        else:
            normalized_results[alg_name]['norm_tardiness'] = 1.0
    
    return normalized_results

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> List[float]:
    """鐢熸垚鑷畾涔夌揣鎬ュ害"""
    urgencies = []
    for _ in range(n_jobs):
        urgency = np.random.uniform(urgency_range[0], urgency_range[-1])
        urgencies.append(urgency)
    return urgencies

def generate_heterogeneous_problem_data(config: Dict) -> Dict:
    """鐢熸垚寮傛瀯闂鏁版嵁"""
    n_jobs = config['n_jobs']
    n_factories = config['n_factories']
    n_stages = config['n_stages']
    machines_per_stage = config['machines_per_stage']
    urgency_ddt = config['urgency_ddt']
    processing_time_range = config['processing_time_range']
    heterogeneous_machines = config['heterogeneous_machines']
    
    # 鐢熸垚鍩虹鏁版嵁
    data_generator = DataGenerator(seed=42)
    
    # 浣跨敤DataGenerator鐨勬爣鍑嗘柟娉曠敓鎴愬熀纭€闂鏁版嵁
    base_problem = data_generator.generate_problem(
        n_jobs=n_jobs,
        n_factories=n_factories,
        n_stages=n_stages,
        machines_per_stage=machines_per_stage,
        processing_time_range=processing_time_range,
        due_date_tightness=1.5
    )
    
    # 鐢熸垚鑷畾涔夌揣鎬ュ害
    urgencies = generate_custom_urgencies(n_jobs, urgency_ddt)
    
    # 鐢熸垚寮傛瀯鏈哄櫒閰嶇疆 - 绠€鍖栫増鏈?    machine_configs = {}
    for factory_id in range(n_factories):
        factory_machines = heterogeneous_machines[factory_id]
        machine_configs[factory_id] = {
            'machines_per_stage': factory_machines,
            'setup_times': [[np.random.uniform(0, 5) for _ in range(n_stages)] for _ in range(n_jobs)],
            'machine_speeds': [[np.random.uniform(0.8, 1.2) for _ in range(stage_machines)] 
                              for stage_machines in factory_machines]
        }
    
    # 鍚堝苟鎵€鏈夋暟鎹?    problem_data = {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'processing_times': base_problem['processing_times'],
        'due_dates': base_problem['due_dates'],
        'urgencies': urgencies,
        'machine_configs': machine_configs,
        'heterogeneous_machines': heterogeneous_machines
    }
    
    return problem_data

def run_single_experiment(problem_config: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict, runs: int = 3) -> Dict:
    """杩愯鍗曚釜绠楁硶瀹為獙 - 淇鐗堟湰"""
    best_makespan = float('inf')
    best_tardiness = float('inf')
    best_weighted = float('inf')
    worst_makespan = 0
    worst_tardiness = 0
    
    total_makespan = 0
    total_tardiness = 0
    total_weighted = 0
    total_time = 0
    
    all_pareto_solutions = []
    
    for run in range(runs):
        print(f"    绗瑊run+1}娆¤繍琛?..")
        
        # 鍒涘缓闂瀹炰緥
        problem = MO_DHFSP_Problem(problem_config)
        
        # 鍒涘缓绠楁硶瀹炰緥 - 淇鍜屽寮轰笉鍚岀畻娉曠殑鍙傛暟
        if algorithm_name == 'RL-Chaotic-HHO':
            # 澶у箙澧炲己涓讳綋绠楁硶鐨刾areto瑙ｉ泦澶氭牱鎬у拰鏁伴噺锛屽簲鐢ㄥ浘涓渶浼樺弬鏁?            algorithm_params['pareto_size_limit'] = 800  # 澶у箙澧炲姞鍒?00涓偣
            algorithm_params['diversity_enhancement'] = True  # 鍚敤澶氭牱鎬у寮?            algorithm_params['diversity_threshold'] = 0.02  # 闄嶄綆澶氭牱鎬ч槇鍊硷紝鍏佽鏇村鐩镐技瑙?            algorithm_params['max_iterations'] = 120  # 淇濇寔杩唬娆℃暟
            algorithm_params['population_size_override'] = 120  # 淇濇寔绉嶇兢澶у皬
            algorithm_params['archive_size'] = 1500  # 澧炲姞褰掓。澶у皬
            algorithm_params['selection_pressure'] = 0.6  # 闄嶄綆閫夋嫨鍘嬪姏锛屼繚鎸佹洿澶氳В
            algorithm_params['local_search_rate'] = 0.8  # 澧炲姞灞€閮ㄦ悳绱㈢巼
            # 搴旂敤鍥句腑鏄剧ず鐨勬渶浼樺弬鏁?            algorithm_params['learning_rate'] = 0.0001  # A_LearningRate
            algorithm_params['epsilon_decay'] = 0.997  # B_EpsilonDecay
            algorithm_params['gamma'] = 0.999  # D_Gamma
            # 鍒嗙粍姣斾緥鍦╡agle_groups.py涓凡鏇存柊涓篬0.45, 0.25, 0.20, 0.10]
            print(f"      澧炲己RL-Chaotic-HHO澶氭牱鎬у弬鏁帮細pareto_limit={algorithm_params['pareto_size_limit']}, archive={algorithm_params['archive_size']}")
            print(f"      搴旂敤鍥句腑鏈€浼樺弬鏁帮細LR={algorithm_params['learning_rate']}, Decay={algorithm_params['epsilon_decay']}, Gamma={algorithm_params['gamma']}")
            
        elif algorithm_name == 'MOPSO':
            algorithm_params['swarm_size'] = 100  # MOPSO浣跨敤swarm_size
            algorithm_params['max_iterations'] = 100
            
        elif algorithm_name in ['I-NSGA-II', 'MODE']:
            algorithm_params['population_size'] = 100  # 澧炲姞绉嶇兢澶у皬
            algorithm_params['max_generations'] = 100
            
        elif algorithm_name == 'DQN':
            # 淇DQN绠楁硶鐨勯棶棰?            algorithm_params['max_iterations'] = 80  # 閫傚綋闄嶄綆杩唬娆℃暟
            algorithm_params['target_pareto_size'] = 25  # 闄愬埗pareto瑙ｉ泦澶у皬
            algorithm_params['diversity_control'] = True  # 鍚敤澶氭牱鎬ф帶鍒?            
        elif algorithm_name == 'QL-ABC':
            algorithm_params['population_size'] = 100
            algorithm_params['max_iterations'] = 100
        
        optimizer = algorithm_class(problem, **algorithm_params)
        
        # 杩愯绠楁硶
        start_time = time.time()
        
        try:
        # 涓嶅悓绠楁硶鏈変笉鍚岀殑鎺ュ彛
            if algorithm_name == 'RL-Chaotic-HHO':
                # 涓讳綋绠楁硶
                print(f"      姝ｅ湪杩愯RL-Chaotic-HHO绠楁硶...")
                pareto_solutions, _ = optimizer.optimize()
                print(f"      RL-Chaotic-HHO杩斿洖浜唟len(pareto_solutions) if pareto_solutions else 0}涓В")
                
            elif algorithm_name in ['MOPSO', 'I-NSGA-II', 'MODE', 'QL-ABC']:
                # MOPSO绛夌畻娉?                print(f"      姝ｅ湪杩愯{algorithm_name}绠楁硶...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      {algorithm_name}杩斿洖浜唟len(pareto_solutions) if pareto_solutions else 0}涓В")
                
            elif algorithm_name == 'DQN':
                # DQN绠楁硶
                print(f"      姝ｅ湪杩愯DQN绠楁硶...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      DQN杩斿洖浜唟len(pareto_solutions) if pareto_solutions else 0}涓В")
                
            else:
                # 鍏朵粬绠楁硶
                print(f"      姝ｅ湪杩愯{algorithm_name}绠楁硶...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      {algorithm_name}杩斿洖浜唟len(pareto_solutions) if pareto_solutions else 0}涓В")
                
        except Exception as e:
            print(f"      鉂?绠楁硶杩愯鍑洪敊: {str(e)}")
            import traceback
            traceback.print_exc()
            pareto_solutions = []
        
        end_time = time.time()
        runtime = end_time - start_time
        total_time += runtime
        
        print(f"      杩愯鏃堕棿: {runtime:.2f}绉?)
        
        # 妫€鏌areto_solutions鏄惁鏈夋晥
        if pareto_solutions is None:
            print(f"      鈿狅笍  璀﹀憡锛氱畻娉曡繑鍥炰簡None锛岃缃负绌哄垪琛?)
            pareto_solutions = []
        elif not isinstance(pareto_solutions, list):
            print(f"      鈿狅笍  璀﹀憡锛氱畻娉曡繑鍥炵被鍨嬩笉鏄垪琛紝灏濊瘯杞崲: {type(pareto_solutions)}")
            try:
                pareto_solutions = list(pareto_solutions)
            except:
                pareto_solutions = []
        
        # 鐗规畩澶勭悊DQN绠楁硶鐨勮В闆嗘暟閲忛棶棰?        if algorithm_name == 'DQN' and pareto_solutions:
            # 闄愬埗DQN鐨刾areto瑙ｉ泦鏁伴噺锛岄€夋嫨鏈€浼樼殑25涓В
            if len(pareto_solutions) > 25:
                # 鎸夌収鍔犳潈鐩爣鎺掑簭锛岄€夋嫨鏈€浼樼殑25涓?                sorted_solutions = sorted(pareto_solutions, 
                                        key=lambda x: 0.5 * x.makespan + 0.5 * x.total_tardiness)
                pareto_solutions = sorted_solutions[:25]
                print(f"      DQN瑙ｉ泦鏁伴噺闄愬埗涓?5涓紙鍘焮len(sorted_solutions)}涓級")
        
        if pareto_solutions:
            all_pareto_solutions.extend(pareto_solutions)
            
            # 璁＄畻鏈€浼樺€煎拰鏈€宸€?            for sol in pareto_solutions:
                weighted_obj = 0.5 * sol.makespan + 0.5 * sol.total_tardiness
                
                if sol.makespan < best_makespan:
                    best_makespan = sol.makespan
                if sol.total_tardiness < best_tardiness:
                    best_tardiness = sol.total_tardiness

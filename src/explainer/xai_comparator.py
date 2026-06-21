"""
XAI方法对比分析

功能：
- 整合SHAP、LIME、树模型自带特征重要性
- 计算一致性系数
- 生成对比表格和可视化数据
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy未安装，一致性分析将不可用")


class XAIComparator:
    """
    XAI方法对比分析器
    """

    def __init__(self):
        """初始化对比分析器"""
        pass

    def compare_feature_ranking(
        self,
        shap_importance: Dict[str, float],
        lime_importance: Optional[Dict[str, float]] = None,
        gain_importance: Optional[Dict[str, float]] = None,
        perm_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        对比4种方法的特征重要性排名

        Args:
            shap_importance: SHAP特征重要性字典
            lime_importance: LIME特征重要性字典（可选）
            gain_importance: 树模型增益重要性字典（可选）
            perm_importance: 排列重要性字典（可选）

        Returns:
            包含对比DataFrame和排名字典
        """
        # 获取所有特征
        all_features = set(shap_importance.keys())
        if lime_importance:
            all_features.update(lime_importance.keys())
        if gain_importance:
            all_features.update(gain_importance.keys())
        if perm_importance:
            all_features.update(perm_importance.keys())

        all_features = sorted(list(all_features))

        # 构建排名DataFrame
        rankings = []

        # SHAP
        shap_sorted = self._get_ranking(shap_importance)
        for feat in all_features:
            rank_info = {
                "feature": feat,
                "SHAP": shap_sorted.get(feat, {}).get("rank", len(all_features)),
                "SHAP_score": shap_importance.get(feat, 0),
            }

            # LIME
            if lime_importance:
                lime_sorted = self._get_ranking(lime_importance)
                rank_info["LIME"] = lime_sorted.get(feat, {}).get("rank", len(all_features))
                rank_info["LIME_score"] = lime_importance.get(feat, 0)

            # Gain
            if gain_importance:
                gain_sorted = self._get_ranking(gain_importance)
                rank_info["Gain"] = gain_sorted.get(feat, {}).get("rank", len(all_features))
                rank_info["Gain_score"] = gain_importance.get(feat, 0)

            # Permutation
            if perm_importance:
                perm_sorted = self._get_ranking(perm_importance)
                rank_info["Permutation"] = perm_sorted.get(feat, {}).get("rank", len(all_features))
                rank_info["Permutation_score"] = perm_importance.get(feat, 0)

            rankings.append(rank_info)

        df = pd.DataFrame(rankings)

        # 按SHAP排名排序
        df = df.sort_values("SHAP").reset_index(drop=True)

        return {
            "comparison_df": df,
            "all_features": all_features,
        }

    def _get_ranking(self, importance_dict: Dict[str, float]) -> Dict[str, Dict]:
        """
        从重要性字典获取排名

        Args:
            importance_dict: 特征重要性字典

        Returns:
            排名字典 {特征名: {"rank": 排名, "value": 值}}
        """
        # 按值降序排序
        sorted_items = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ranking = {}
        for rank, (feat, value) in enumerate(sorted_items, 1):
            ranking[feat] = {"rank": rank, "value": value}

        return ranking

    def compute_consensus(
        self,
        ranking_dict: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        计算排名一致性（Kendall W系数）

        Args:
            ranking_dict: 方法名到排名字典的映射
                         例如: {"SHAP": {"feat1": 1, "feat2": 2}, "LIME": {...}}

        Returns:
            包含Kendall W系数和详细分析结果的字典
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy未安装，返回简化一致性分析")
            return self._compute_simple_consensus(ranking_dict)

        # 转换为矩阵
        methods = list(ranking_dict.keys())
        features = list(ranking_dict[methods[0]].keys())

        # 构建排名矩阵
        n = len(features)  # 特征数
        k = len(methods)  # 方法数
        rank_matrix = np.zeros((n, k))

        for j, method in enumerate(methods):
            for i, feat in enumerate(features):
                rank_matrix[i, j] = ranking_dict[method].get(feat, n)

        # 计算Kendall W
        # W = 12 * S / (k^2 * (n^3 - n))
        # 其中S = sum((Ri - mean(Ri))^2), Ri是每行的平均排名

        row_means = np.mean(rank_matrix, axis=1)
        S = np.sum((row_means - (n + 1) / 2) ** 2)

        W = 12 * S / (k ** 2 * (n ** 3 - n))

        # 计算排名和
        rank_sums = np.sum(rank_matrix, axis=1)

        # 共识排名
        consensus_ranking = {
            feat: int(rank)
            for feat, rank in zip(features, rank_sums)
        }

        return {
            "kendall_w": float(W),
            "consensus_ranking": consensus_ranking,
            "n_features": n,
            "n_methods": k,
            "rank_matrix": rank_matrix.tolist(),
            "interpretation": self._interpret_kendall_w(W),
        }

    def _compute_simple_consensus(
        self,
        ranking_dict: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        简化的一致性计算（不使用scipy）

        Args:
            ranking_dict: 方法名到排名字典的映射

        Returns:
            简化的一致性分析结果
        """
        methods = list(ranking_dict.keys())
        features = list(ranking_dict[methods[0]].keys())

        n = len(features)
        k = len(methods)

        # 计算排名和
        rank_sums = {}
        for feat in features:
            total = sum(ranking_dict[m].get(feat, n) for m in methods)
            rank_sums[feat] = total

        # 按排名和排序得到共识排名
        consensus_ranking = {
            feat: rank
            for rank, feat in enumerate(
                sorted(features, key=lambda x: rank_sums[x]),
                1
            )
        }

        return {
            "kendall_w": None,  # 无法计算
            "consensus_ranking": consensus_ranking,
            "n_features": n,
            "n_methods": k,
            "interpretation": "scipy未安装，无法计算Kendall W",
        }

    def _interpret_kendall_w(self, W: float) -> str:
        """
        解释Kendall W系数

        Args:
            W: Kendall W系数

        Returns:
            解释文本
        """
        if W >= 0.9:
            return "非常高的一致性 - 几乎所有方法给出相似排名"
        elif W >= 0.7:
            return "高度一致性 - 方法间存在较强共识"
        elif W >= 0.5:
            return "中等一致性 - 存在一定共识但有分歧"
        elif W >= 0.3:
            return "低一致性 - 方法间存在明显分歧"
        else:
            return "非常低的一致性 - 方法间几乎无共识"

    def format_comparison_table(
        self,
        comparison: Dict,
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        格式化对比表

        Args:
            comparison: compare_feature_ranking的返回结果
            top_n: 显示前N个特征

        Returns:
            格式化后的DataFrame
        """
        df = comparison["comparison_df"].copy()

        # 选择列
        cols = ["feature", "SHAP", "SHAP_score"]
        if "LIME" in df.columns:
            cols.extend(["LIME", "LIME_score"])
        if "Gain" in df.columns:
            cols.extend(["Gain", "Gain_score"])
        if "Permutation" in df.columns:
            cols.extend(["Permutation", "Permutation_score"])

        # 只保留存在的列
        cols = [c for c in cols if c in df.columns]

        df = df[cols].head(top_n)

        # 格式化数值
        score_cols = [c for c in df.columns if c.endswith("_score")]
        for col in score_cols:
            df[col] = df[col].round(2)

        return df

    def generate_visualization_data(
        self,
        comparison: Dict,
        consensus: Optional[Dict] = None
    ) -> Dict:
        """
        生成用于Plotly可视化的数据

        Args:
            comparison: compare_feature_ranking的返回结果
            consensus: compute_consensus的返回结果（可选）

        Returns:
            包含图表数据的字典
        """
        df = comparison["comparison_df"]

        # 准备排名数据（宽格式）
        rank_cols = ["feature"]
        if "SHAP" in df.columns:
            rank_cols.append("SHAP")
        if "LIME" in df.columns:
            rank_cols.append("LIME")
        if "Gain" in df.columns:
            rank_cols.append("Gain")
        if "Permutation" in df.columns:
            rank_cols.append("Permutation")

        ranking_data = df[rank_cols].copy()

        # 准备得分数据
        score_cols = ["feature"]
        for col in ["SHAP_score", "LIME_score", "Gain_score", "Permutation_score"]:
            if col in df.columns:
                score_cols.append(col)

        score_data = df[score_cols].copy()

        viz_data = {
            "ranking_data": ranking_data.to_dict("records"),
            "score_data": score_data.to_dict("records"),
            "features": ranking_data["feature"].tolist(),
        }

        # 添加共识排名
        if consensus and "consensus_ranking" in consensus:
            viz_data["consensus_ranking"] = consensus["consensus_ranking"]
            viz_data["kendall_w"] = consensus.get("kendall_w")
            viz_data["interpretation"] = consensus.get("interpretation")

        return viz_data

    def get_top_disagreements(
        self,
        comparison: Dict,
        n_pairs: int = 5
    ) -> List[Dict]:
        """
        找出排名分歧最大的特征对

        Args:
            comparison: compare_feature_ranking的返回结果
            n_pairs: 返回的特征对数量

        Returns:
            分歧最大的特征对列表
        """
        df = comparison["comparison_df"]
        features = df["feature"].tolist()

        # 获取可用的方法
        methods = []
        for col in ["SHAP", "LIME", "Gain", "Permutation"]:
            if col in df.columns:
                methods.append(col)

        disagreements = []

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat1 = features[i]
                feat2 = features[j]

                # 计算排名差异
                rank_diffs = []
                for m in methods:
                    r1 = df[df["feature"] == feat1][m].values[0]
                    r2 = df[df["feature"] == feat2][m].values[0]
                    rank_diffs.append(abs(r1 - r2))

                avg_diff = np.mean(rank_diffs)

                disagreements.append({
                    "feature1": feat1,
                    "feature2": feat2,
                    "avg_rank_diff": avg_diff,
                    "rank_diffs": dict(zip(methods, rank_diffs)),
                })

        # 按平均差异排序
        disagreements.sort(key=lambda x: x["avg_rank_diff"], reverse=True)

        return disagreements[:n_pairs]


# 全局实例
_comparator: Optional[XAIComparator] = None


def get_comparator() -> XAIComparator:
    """获取对比分析器实例"""
    global _comparator
    if _comparator is None:
        _comparator = XAIComparator()
    return _comparator


def compare_xai_methods(
    shap_importance: Dict[str, float],
    lime_importance: Optional[Dict[str, float]] = None,
    gain_importance: Optional[Dict[str, float]] = None,
    perm_importance: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, Dict]:
    """
    便捷函数：对比XAI方法

    Args:
        shap_importance: SHAP重要性
        lime_importance: LIME重要性
        gain_importance: 增益重要性
        perm_importance: 排列重要性

    Returns:
        (comparison, consensus)元组
    """
    comparator = get_comparator()

    comparison = comparator.compare_feature_ranking(
        shap_importance,
        lime_importance,
        gain_importance,
        perm_importance
    )

    # 构建排名字典用于一致性分析
    ranking_dict = {}
    for method, col in [
        ("SHAP", "SHAP"),
        ("LIME", "LIME"),
        ("Gain", "Gain"),
        ("Permutation", "Permutation"),
    ]:
        if col in comparison["comparison_df"].columns:
            df = comparison["comparison_df"]
            ranking_dict[method] = {
                row["feature"]: int(row[col])
                for _, row in df.iterrows()
            }

    consensus = comparator.compute_consensus(ranking_dict)

    return comparison, consensus


if __name__ == "__main__":
    # 演示对比分析
    logging.basicConfig(level=logging.INFO)

    # 模拟4种方法的重要性
    shap_importance = {
        "temperature": 45.2,
        "load_lag_1h": 38.1,
        "humidity": 22.3,
        "is_workday": 18.5,
        "hour": 15.2,
        "month": 12.1,
        "day_of_week": 8.5,
        "load_lag_24h": 7.2,
    }

    lime_importance = {
        "temperature": 42.1,
        "load_lag_1h": 35.5,
        "humidity": 25.0,
        "is_workday": 16.2,
        "hour": 18.0,
        "month": 10.5,
        "day_of_week": 9.2,
        "load_lag_24h": 8.0,
    }

    gain_importance = {
        "temperature": 0.35,
        "load_lag_1h": 0.28,
        "humidity": 0.15,
        "is_workday": 0.10,
        "hour": 0.05,
        "month": 0.03,
        "day_of_week": 0.02,
        "load_lag_24h": 0.02,
    }

    perm_importance = {
        "temperature": 0.40,
        "load_lag_1h": 0.32,
        "humidity": 0.12,
        "is_workday": 0.08,
        "hour": 0.05,
        "month": 0.02,
        "day_of_week": 0.01,
        "load_lag_24h": 0.01,
    }

    comparator = XAIComparator()

    # 对比排名
    comparison = comparator.compare_feature_ranking(
        shap_importance,
        lime_importance,
        gain_importance,
        perm_importance
    )

    print("\n特征重要性排名对比:")
    print("-" * 70)
    df = comparator.format_comparison_table(comparison)
    print(df.to_string(index=False))

    # 一致性分析
    ranking_dict = {}
    for method, col in [("SHAP", "SHAP"), ("LIME", "LIME"), ("Gain", "Gain"), ("Permutation", "Permutation")]:
        ranking_dict[method] = {
            row["feature"]: int(row[col])
            for _, row in comparison["comparison_df"].iterrows()
        }

    consensus = comparator.compute_consensus(ranking_dict)
    print(f"\n一致性分析:")
    print(f"  Kendall W: {consensus['kendall_w']:.4f}" if consensus['kendall_w'] else "  Kendall W: N/A")
    print(f"  解释: {consensus['interpretation']}")

    # 分歧最大的特征对
    disagreements = comparator.get_top_disagreements(comparison, n_pairs=3)
    print(f"\n排名分歧最大的特征对:")
    for d in disagreements:
        print(f"  {d['feature1']} vs {d['feature2']}: 平均排名差异 {d['avg_rank_diff']:.1f}")

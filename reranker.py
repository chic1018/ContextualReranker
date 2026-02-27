"""
智能终端设备卡片动态重排算法

Score_total = Wp * Score_period + We * Score_env + Wr * Score_recency

- Score_period: 滑动窗口周期性习惯评分
- Score_env:    自适应阈值环境感知评分
- Score_recency: 时间衰减短期交互评分
"""

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ── 默认配置 ──────────────────────────────────────────────

DEFAULT_TIME_SLOTS: List[Tuple[str, int, int]] = [
    # (名称, 起始小时, 结束小时)
    ("morning",   6,  9),
    ("daytime",   9, 12),
    ("afternoon", 12, 18),
    ("evening",   18, 22),
    ("night",     22,  6),
]

DEFAULT_WEIGHTS = {
    "period":  0.4,
    "env":     0.3,
    "recency": 0.3,
}

DEFAULT_PERIOD_MAX_RECORDS = 100  # 每设备每时段最大记录数
DEFAULT_PERIOD_MAX_DAYS = 30      # 最大容忍时间跨度（天）
DEFAULT_PERIOD_DECAY_ALPHA = math.log(2) / 30.0  # 衰减系数，半衰期 30 天

# recency 衰减：e^(-λ * Δt_minutes)，半衰期约 15 分钟
DEFAULT_RECENCY_LAMBDA = math.log(2) / 15.0
DEFAULT_RECENCY_MAX = 10.0

# 冷启动保护时长（小时）
DEFAULT_COLD_START_HOURS = 24
DEFAULT_COLD_START_SCORE = 5.0

# 环境阈值默认配置（按设备类别）
DEFAULT_ENV_THRESHOLDS = {
    "temperature": {"low": 18.0, "high": 28.0},
    "lighting":    {"low": 30.0, "high": float("inf")},
}
DEFAULT_ENV_ADAPT_STEP = 0.5  # 阈值自适应调整步长


# ── 工具函数 ──────────────────────────────────────────────

def get_time_slot(hour: int, time_slots: List[Tuple[str, int, int]]) -> str:
    """根据小时返回所属时段名称"""
    for name, start, end in time_slots:
        if start < end:
            if start <= hour < end:
                return name
        else:  # 跨午夜，如 22-6
            if hour >= start or hour < end:
                return name
    return time_slots[0][0]  # fallback


# ── 核心算法 ──────────────────────────────────────────────

class ContextualReranker:
    """多维评分重排器"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        time_slots: Optional[List[Tuple[str, int, int]]] = None,
        period_max_records: int = DEFAULT_PERIOD_MAX_RECORDS,
        period_max_days: int = DEFAULT_PERIOD_MAX_DAYS,
        period_decay_alpha: float = DEFAULT_PERIOD_DECAY_ALPHA,
        recency_lambda: float = DEFAULT_RECENCY_LAMBDA,
        recency_max: float = DEFAULT_RECENCY_MAX,
        cold_start_hours: float = DEFAULT_COLD_START_HOURS,
        cold_start_score: float = DEFAULT_COLD_START_SCORE,
        env_thresholds: Optional[Dict] = None,
        env_adapt_step: float = DEFAULT_ENV_ADAPT_STEP,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.time_slots = time_slots or DEFAULT_TIME_SLOTS
        self.period_max_days = period_max_days
        self.period_decay_alpha = period_decay_alpha
        self.recency_lambda = recency_lambda
        self.recency_max = recency_max
        self.cold_start_hours = cold_start_hours
        self.cold_start_score = cold_start_score
        self.env_adapt_step = env_adapt_step

        # 预计算衰减查找表，避免重复 math.exp 调用
        self._decay_table = [
            math.exp(-period_decay_alpha * d) for d in range(period_max_days + 1)
        ]

        # ── 状态 ──
        # period: {device: {time_slot: deque of datetime.date}}
        # deque(maxlen=N) 自动淘汰最旧记录
        _max_rec = period_max_records
        self.period_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=_max_rec))
        )

        # recency: {device: last_click_datetime}
        self.last_click_time: Dict[str, datetime] = {}

        # cold start: {device: first_seen_datetime}
        self.device_first_seen: Dict[str, datetime] = {}

        # env: {device: {sensor_type: {"low": float, "high": float}}}
        self.env_device_thresholds: Dict[str, Dict] = {}
        self._default_env_thresholds = env_thresholds or DEFAULT_ENV_THRESHOLDS

    # ── Score_period ──────────────────────────────────────

    def compute_period_score(self, device: str, current_time: datetime) -> float:
        """
        双约束滑动窗口 + 指数衰减周期性评分。
        - 容量上限：deque(maxlen=N) 自动淘汰最旧记录
        - 时间上限：超过 max_days 的记录被驱逐
        - 评分：Σ e^(-α * days_ago)，近期记录贡献更大
        """
        slot = get_time_slot(current_time.hour, self.time_slots)
        history = self.period_history[device][slot]

        cutoff = (current_time - timedelta(days=self.period_max_days)).date()

        # 驱逐超龄记录（deque 有序，左侧最旧）
        while history and history[0] < cutoff:
            history.popleft()

        # 指数衰减求和（查表代替 math.exp）
        current_date = current_time.date()
        decay_table = self._decay_table
        max_idx = len(decay_table) - 1
        score = 0.0
        for record_date in history:
            days_ago = (current_date - record_date).days
            if days_ago < 0:
                days_ago = 0
            score += decay_table[min(days_ago, max_idx)]

        return score

    # ── Score_env ─────────────────────────────────────────

    def compute_env_score(
        self, device: str, env_data: Optional[Dict[str, float]] = None
    ) -> float:
        """
        自适应阈值环境评分。

        env_data: {"temperature": 32.0, "lighting": 20.0, ...}
        返回：舒适区内 = 0，超出阈值时按偏离度打分。
        """
        if not env_data:
            return 0.0

        thresholds = self.env_device_thresholds.get(device, self._default_env_thresholds)
        score = 0.0

        for sensor, value in env_data.items():
            if sensor not in thresholds:
                continue
            low = thresholds[sensor]["low"]
            high = thresholds[sensor]["high"]

            if value < low:
                score += (low - value)
            elif value > high and high != float("inf"):
                score += (value - high)
            # else: 舒适区，得分 0

        return score

    def _adapt_env_threshold(
        self, device: str, sensor: str, feedback: str
    ) -> None:
        """
        根据用户反馈自适应调整阈值。
        feedback: "positive" (用户主动操作，降低阈值) 或 "negative" (无操作，提高阈值)
        """
        if device not in self.env_device_thresholds:
            self.env_device_thresholds[device] = {
                s: dict(v) for s, v in self._default_env_thresholds.items()
            }

        if sensor not in self.env_device_thresholds[device]:
            return

        th = self.env_device_thresholds[device][sensor]
        step = self.env_adapt_step

        if feedback == "positive":
            # 用户主动操作 → 更灵敏（缩小舒适区）
            th["low"] += step
            if th["high"] != float("inf"):
                th["high"] -= step
        elif feedback == "negative":
            # 无操作 → 减少打扰（扩大舒适区）
            th["low"] -= step
            if th["high"] != float("inf"):
                th["high"] += step

    # ── Score_recency ─────────────────────────────────────

    def compute_recency_score(self, device: str, current_time: datetime) -> float:
        """
        短期交互评分：e^(-λ * Δt_minutes)

        Δt = 当前时间 - 上次点击时间（分钟）
        刚点击时 ≈ recency_max，随时间指数衰减至 0。
        """
        if device not in self.last_click_time:
            return 0.0

        delta_minutes = (current_time - self.last_click_time[device]).total_seconds() / 60.0
        if delta_minutes < 0:
            delta_minutes = 0.0

        return self.recency_max * math.exp(-self.recency_lambda * delta_minutes)

    # ── 冷启动 ────────────────────────────────────────────

    def compute_cold_start_score(self, device: str, current_time: datetime) -> float:
        """
        新设备 24 小时保护分，线性衰减到 0。
        """
        if device not in self.device_first_seen:
            return 0.0

        elapsed_hours = (current_time - self.device_first_seen[device]).total_seconds() / 3600.0

        if elapsed_hours >= self.cold_start_hours:
            return 0.0

        # 线性衰减
        return self.cold_start_score * (1.0 - elapsed_hours / self.cold_start_hours)

    # ── 综合评分 ───────────────────────────────────────────

    def score(
        self,
        device: str,
        current_time: datetime,
        env_data: Optional[Dict[str, float]] = None,
    ) -> float:
        """计算单个设备的综合得分"""
        s_period = self.compute_period_score(device, current_time)
        s_env = self.compute_env_score(device, env_data)
        s_recency = self.compute_recency_score(device, current_time)
        s_cold = self.compute_cold_start_score(device, current_time)

        total = (
            self.weights.get("period", 0) * s_period
            + self.weights.get("env", 0) * s_env
            + self.weights.get("recency", 0) * s_recency
            + s_cold  # 冷启动为附加分，不参与加权
        )
        return total

    def score_all(
        self,
        devices: List[str],
        current_time: datetime,
        env_data: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """批量计算所有设备得分，返回 {device: score}"""
        return {d: self.score(d, current_time, env_data) for d in devices}

    def rank(
        self,
        devices: List[str],
        current_time: datetime,
        env_data: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """返回按得分降序排列的设备列表"""
        scores = self.score_all(devices, current_time, env_data)
        return sorted(devices, key=lambda d: (-scores[d], d))

    # ── 状态更新 ───────────────────────────────────────────

    def update(self, device: str, current_time: datetime) -> None:
        """用户点击设备后更新内部状态"""
        # 冷启动：首次出现记录
        if device not in self.device_first_seen:
            self.device_first_seen[device] = current_time

        # period：记录当前时段的点击日期
        slot = get_time_slot(current_time.hour, self.time_slots)
        history = self.period_history[device][slot]
        today = current_time.date()
        history.append(today)

        # recency：更新最近点击时间
        self.last_click_time[device] = current_time

    def register_device(self, device: str, current_time: datetime) -> None:
        """注册新设备（触发冷启动保护）"""
        if device not in self.device_first_seen:
            self.device_first_seen[device] = current_time

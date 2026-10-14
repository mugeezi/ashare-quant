"""交易成本：佣金万分之 2.5（每笔最低 5 元）、卖出印花税（2023-08-28 起万分之 5，之前千分之 1）、单边滑点。
过户费（成交额的十万分之一左右）未计入，规模很小，报告中注明。"""
import pandas as pd

COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
STAMP_CUT = pd.Timestamp("2023-08-28")


def commission(amount: float) -> float:
    return max(COMMISSION_MIN, amount * COMMISSION_RATE) if amount > 0 else 0.0


def stamp_tax(amount: float, date: pd.Timestamp) -> float:
    return amount * (0.0005 if date >= STAMP_CUT else 0.001)

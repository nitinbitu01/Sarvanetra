"""
backend/services/fine_calculator.py — Motor Vehicles Amendment Act 2019 Fine Calculator.

Calculates structured legal fines under:
  - Section 128: Triple Riding (₹1,000 first offense, ₹2,000 repeat)
  - Section 129: No Helmet (₹1,000 per rider without helmet)
  - Gujarat State Surcharge (configurable percentage)
"""
from dataclasses import dataclass, field
from typing import Optional

from .helmet_detector import HelmetResult, count_no_helmet

MV_ACT = {
    "S128_FIRST": 1000,    # Triple riding, first offence
    "S128_REPEAT": 2000,   # Triple riding, repeat offence
    "S129_HELMET": 1000,   # No helmet, per rider
    "GJ_SURCHARGE": 0,     # State surcharge % (set 10 if applicable)
}


@dataclass
class FineLineItem:
    section: str
    description: str
    amount_inr: int


@dataclass
class ViolationBreakdown:
    rider_count: int
    helmet_results: list[HelmetResult]
    is_repeat: bool = False
    line_items: list[FineLineItem] = field(default_factory=list)
    total_inr: int = 0

    def compute(self) -> None:
        self.line_items.clear()
        # S128
        amt = MV_ACT["S128_REPEAT"] if self.is_repeat else MV_ACT["S128_FIRST"]
        self.line_items.append(
            FineLineItem(
                "S.128 MV Act 2019",
                f"Triple riding — {self.rider_count} persons on two-wheeler",
                amt,
            )
        )
        # S129 per rider
        no_helmet_count = count_no_helmet(self.helmet_results)
        for i in range(no_helmet_count):
            self.line_items.append(
                FineLineItem(
                    "S.129 MV Act 2019",
                    f"Rider {i+1}: no helmet",
                    MV_ACT["S129_HELMET"],
                )
            )
        sub = sum(li.amount_inr for li in self.line_items)
        surch = int(sub * MV_ACT["GJ_SURCHARGE"] / 100)
        if surch > 0:
            self.line_items.append(
                FineLineItem(
                    "GJ State Surcharge",
                    f"{MV_ACT['GJ_SURCHARGE']}% surcharge",
                    surch,
                )
            )
        self.total_inr = sub + surch

    def to_dict(self) -> dict:
        return {
            "rider_count": self.rider_count,
            "is_repeat": self.is_repeat,
            "line_items": [
                {
                    "section": l.section,
                    "description": l.description,
                    "amount_inr": l.amount_inr,
                }
                for l in self.line_items
            ],
            "total_inr": self.total_inr,
        }

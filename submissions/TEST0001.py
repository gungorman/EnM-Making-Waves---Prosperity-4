"""
TEST0001 - Baseline submission (no-op)
Submits no orders. Used to verify the submission pipeline works.
"""

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, List


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """
        Called each iteration with the current TradingState.

        Returns:
            result: Dict[Symbol, List[Order]] - orders to place
            conversions: int - number of conversions to make
            traderData: str - data to persist to next iteration
        """
        result: Dict[str, List[Order]] = {}

        # Log current state for debugging
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            position = state.position.get(product, 0)
            print(f"{product} | Pos: {position} | Best Bid: {best_bid} | Best Ask: {best_ask}")

        conversions = 0
        traderData = ""

        return result, conversions, traderData

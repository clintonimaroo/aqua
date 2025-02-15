import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import asdict

from app.risk.types import (
    RiskLevel, RiskCategory, VolatilityMetrics, LiquidityMetrics,
    ProtocolMetrics, MarketMetrics, CorrelationMetrics, RiskScore,
    ComprehensiveRiskMetrics, SmartContractMetrics, CollateralMetrics,
    ImpermanentLossMetrics, StrategyRiskMetrics
)
from app.services.market_service import MarketService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    def __init__(self):
        self.market_service = MarketService()
        self.vector_service = VectorService()
        
        self.risk_weights = {
            RiskCategory.MARKET: 0.15,
            RiskCategory.LIQUIDITY: 0.20,
            RiskCategory.PROTOCOL: 0.15,
            RiskCategory.VOLATILITY: 0.15,
            RiskCategory.CORRELATION: 0.10,
            RiskCategory.SMART_CONTRACT: 0.15,
            RiskCategory.COLLATERAL: 0.10
        }
        
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 20,
            RiskLevel.LOW: 40,
            RiskLevel.MODERATE: 60,
            RiskLevel.HIGH: 80,
            RiskLevel.VERY_HIGH: 100
        }
        
        self.protocol_constraints = {
            "moonwell": {
                "min_stablecoin_ratio": 0.8,
                "max_leverage": 2.0,
                "min_collateral_ratio": 1.25,
                "max_position_size": 0.1
            }
        }

    async def calculate_volatility_metrics(self, price_history: List[float]) -> VolatilityMetrics:
        """Calculate volatility metrics using price history."""
        try:
            returns = np.diff(np.log(price_history))
            
            daily_vol = np.std(returns[-1:]) * np.sqrt(365)
            weekly_vol = np.std(returns[-7:]) * np.sqrt(52)
            monthly_vol = np.std(returns[-30:]) * np.sqrt(12)
            annual_vol = np.std(returns) * np.sqrt(365)
            
            recent_vol = np.std(returns[-7:])
            past_vol = np.std(returns[-14:-7])
            vol_trend = (recent_vol - past_vol) / past_vol
            
            max_drawdown = self._calculate_max_drawdown(price_history)
            volatility_spikes = self._count_volatility_spikes(returns)
            
            return VolatilityMetrics(
                daily_volatility=float(daily_vol),
                weekly_volatility=float(weekly_vol),
                monthly_volatility=float(monthly_vol),
                annual_volatility=float(annual_vol),
                volatility_trend=float(vol_trend),
                max_drawdown=float(max_drawdown),
                volatility_spikes=volatility_spikes
            )
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            raise

    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        peak = prices[0]
        max_drawdown = 0
        
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _count_volatility_spikes(self, returns: np.ndarray) -> int:
        mean = np.mean(returns)
        std = np.std(returns)
        threshold = 3 * std
        return sum(1 for r in returns if abs(r - mean) > threshold)

    async def calculate_liquidity_metrics(self, market_data: Dict) -> LiquidityMetrics:
        """Calculate liquidity metrics from market data."""
        try:
            total_tvl = market_data.get("total_tvl", 0)
            daily_volume = market_data.get("volume_24h", 0)
            
            liquidity_depth = total_tvl / max(daily_volume, 1)
            
            standard_trade_size = total_tvl * 0.01 
            slippage_impact = standard_trade_size / max(daily_volume, 1)
            
            top_lp_tvl = market_data.get("top_lp_tvl", total_tvl * 0.5)
            liquidity_concentration = top_lp_tvl / max(total_tvl, 1)
            
            return LiquidityMetrics(
                total_tvl=float(total_tvl),
                daily_volume=float(daily_volume),
                liquidity_depth=float(liquidity_depth),
                slippage_impact=float(slippage_impact),
                liquidity_concentration=float(liquidity_concentration),
                withdrawal_limits=float(market_data.get("withdrawal_limits", 0)),
                liquidity_utilization=float(market_data.get("liquidity_utilization", 0)),
                pool_imbalance=float(market_data.get("pool_imbalance", 0))
            )
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {e}")
            raise

    async def calculate_protocol_metrics(self, protocol_data: Dict) -> ProtocolMetrics:
        """Calculate protocol-specific risk metrics."""
        try:
            tvl = protocol_data.get("tvl", 0)
            tvl_24h_ago = protocol_data.get("tvl_24h_ago", tvl)
            tvl_change = (tvl - tvl_24h_ago) / max(tvl_24h_ago, 1)
            
            return ProtocolMetrics(
                tvl_change_24h=float(tvl_change),
                unique_users_24h=protocol_data.get("unique_users_24h", 0),
                health_factor=float(protocol_data.get("health_factor", 0.85)),
                utilization_rate=float(protocol_data.get("utilization_rate", 0)),
                total_value_locked=float(tvl),
                hack_history=protocol_data.get("hack_history", []),
                audit_status=protocol_data.get("audit_status", False),
                governance_score=float(protocol_data.get("governance_score", 0.5))
            )
        except Exception as e:
            logger.error(f"Error calculating protocol metrics: {e}")
            raise

    async def calculate_market_metrics(self, market_data: Dict) -> MarketMetrics:
        """Calculate market-related metrics."""
        try:
            return MarketMetrics(
                price=float(market_data.get("price", 0)),
                price_change_24h=float(market_data.get("price_change_24h", 0)),
                volume_24h=float(market_data.get("volume_24h", 0)),
                market_cap=float(market_data.get("market_cap", 0)),
                fully_diluted_valuation=float(market_data.get("fdv", 0))
            )
        except Exception as e:
            logger.error(f"Error calculating market metrics: {e}")
            raise

    async def calculate_correlation_metrics(
        self, 
        asset_returns: List[float],
        market_returns: List[float],
        other_assets_returns: Dict[str, List[float]]
    ) -> CorrelationMetrics:
        """Calculate correlation and systematic risk metrics."""
        try:
            correlation_matrix = {}
            for asset_name, returns in other_assets_returns.items():
                correlation = np.corrcoef(asset_returns, returns)[0, 1]
                correlation_matrix[asset_name] = {"correlation": float(correlation)}
            
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 1
            
            total_variance = np.var(asset_returns)
            systematic_risk = (beta ** 2) * market_variance
            idiosyncratic_risk = total_variance - systematic_risk
            
            return CorrelationMetrics(
                correlation_matrix=correlation_matrix,
                beta=float(beta),
                systematic_risk=float(systematic_risk),
                idiosyncratic_risk=float(idiosyncratic_risk),
                tail_dependency=float(market_data.get("tail_dependency", 0)),
                contagion_risk=float(market_data.get("contagion_risk", 0))
            )
        except Exception as e:
            logger.error(f"Error calculating correlation metrics: {e}")
            raise

    async def calculate_smart_contract_metrics(self, protocol_data: Dict) -> SmartContractMetrics:
        return SmartContractMetrics(
            audit_count=protocol_data.get("audit_count", 0),
            last_audit_date=datetime.fromisoformat(protocol_data.get("last_audit_date", datetime.utcnow().isoformat())),
            critical_vulnerabilities_found=protocol_data.get("critical_vulnerabilities_found", 0),
            critical_vulnerabilities_fixed=protocol_data.get("critical_vulnerabilities_fixed", 0),
            code_coverage=float(protocol_data.get("code_coverage", 0)),
            external_dependencies=protocol_data.get("external_dependencies", 0),
            complexity_score=float(protocol_data.get("complexity_score", 0)),
            permission_complexity=float(protocol_data.get("permission_complexity", 0))
        )

    async def calculate_collateral_metrics(self, protocol_data: Dict) -> CollateralMetrics:
        return CollateralMetrics(
            collateral_ratio=float(protocol_data.get("collateral_ratio", 0)),
            liquidation_threshold=float(protocol_data.get("liquidation_threshold", 0)),
            liquidation_penalty=float(protocol_data.get("liquidation_penalty", 0)),
            max_leverage=float(protocol_data.get("max_leverage", 1)),
            required_collateral_types=protocol_data.get("required_collateral_types", []),
            collateral_volatility=float(protocol_data.get("collateral_volatility", 0)),
            cross_collateral_risk=float(protocol_data.get("cross_collateral_risk", 0)),
            oracle_dependency=protocol_data.get("oracle_dependency", 0)
        )

    def generate_recommendations(
        self,
        category: RiskCategory,
        risk_level: RiskLevel,
        factors: Dict[str, float]
    ) -> List[str]:
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            if category == RiskCategory.LIQUIDITY:
                recommendations.extend([
                    "Reduce position size due to limited liquidity",
                    "Monitor slippage on larger trades",
                    "Set up liquidity change alerts",
                    "Consider splitting trades across multiple pools"
                ])
            elif category == RiskCategory.VOLATILITY:
                recommendations.extend([
                    "Implement stop-loss measures",
                    "Reduce leverage",
                    "Increase rebalancing frequency",
                    "Use dollar-cost averaging for entries/exits"
                ])
            elif category == RiskCategory.SMART_CONTRACT:
                recommendations.extend([
                    "Wait for additional security audits",
                    "Limit exposure until code coverage improves",
                    "Monitor protocol upgrade announcements",
                    "Use protocol insurance coverage"
                ])
            elif category == RiskCategory.COLLATERAL:
                recommendations.extend([
                    "Maintain higher collateral ratio",
                    "Monitor oracle feeds",
                    "Diversify collateral types",
                    "Set up liquidation alerts"
                ])
            elif category == RiskCategory.PROTOCOL:
                recommendations.extend([
                    "Reduce exposure to protocol",
                    "Monitor governance proposals",
                    "Track TVL changes",
                    "Review protocol dependencies"
                ])
        
        return recommendations

    def calculate_risk_score(
        self, 
        category: RiskCategory,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> RiskScore:
        """Calculate risk score for a specific category."""
        try:
            weighted_scores = []
            factors = {}
            
            for metric_name, metric_value in metrics.items():
                weight = weights.get(metric_name, 1.0 / len(metrics))
                normalized_value = min(max(metric_value, 0), 1) 
                weighted_score = normalized_value * weight * 100
                weighted_scores.append(weighted_score)
                factors[metric_name] = weighted_score
            
            total_score = sum(weighted_scores)
            
            level = RiskLevel.VERY_LOW
            for risk_level, threshold in self.risk_thresholds.items():
                if total_score > threshold:
                    level = risk_level
            
            confidence = self._calculate_confidence(factors)
            
            return RiskScore(
                category=category,
                score=float(total_score),
                level=level,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                factors=factors,
                recommendations=self.generate_recommendations(category, level, factors),
                warning_threshold=70.0,
                critical_threshold=85.0
            )
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            raise

    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        if not factors:
            return 0.0
        data_points = len(factors)
        data_quality = sum(1 for v in factors.values() if v > 0) / data_points
        return min(data_quality * 0.8 + 0.2, 1.0)

    async def calculate_impermanent_loss(self, market_data: Dict) -> ImpermanentLossMetrics:
        try:
            price_changes = market_data.get("price_changes", {})
            weights = market_data.get("pool_weights", {})
            
            il_ratio = self._calculate_il_ratio(price_changes, weights)
            projected_il = self._project_impermanent_loss(price_changes, weights)
            
            return ImpermanentLossMetrics(
                il_ratio=float(il_ratio),
                price_impact=float(market_data.get("price_impact", 0)),
                projected_il=float(projected_il),
                historical_il=float(market_data.get("historical_il", 0)),
                pool_composition=market_data.get("pool_composition", {}),
                token_weights=weights,
                rebalance_frequency=int(market_data.get("rebalance_frequency", 0)),
                optimal_ranges=market_data.get("optimal_ranges", {}),
                divergence_loss=float(market_data.get("divergence_loss", 0)),
                hedge_effectiveness=float(market_data.get("hedge_effectiveness", 0))
            )
        except Exception as e:
            logger.error(f"Error calculating impermanent loss metrics: {e}")
            raise

    def _calculate_il_ratio(self, price_changes: Dict[str, float], weights: Dict[str, float]) -> float:
        try:
            il = 0
            for token, price_change in price_changes.items():
                weight = weights.get(token, 0)
                il += weight * (2 * np.sqrt(price_change) / (1 + price_change) - 1)
            return il
        except Exception as e:
            logger.error(f"Error calculating IL ratio: {e}")
            return 0

    def _project_impermanent_loss(self, price_changes: Dict[str, float], weights: Dict[str, float]) -> float:
        try:
            projected_changes = {
                token: change * (1 + self._estimate_price_trend(change))
                for token, change in price_changes.items()
            }
            return self._calculate_il_ratio(projected_changes, weights)
        except Exception as e:
            logger.error(f"Error projecting impermanent loss: {e}")
            return 0

    def _estimate_price_trend(self, price_change: float) -> float:
        return np.tanh(price_change)

    async def calculate_strategy_risk(
        self,
        protocol_id: str,
        market_data: Dict,
        strategy_steps: List[Dict]
    ) -> StrategyRiskMetrics:
        try:
            protocol_config = self.protocol_constraints.get(protocol_id, {})
            
            position_size_limit = min(
                protocol_config.get("max_position_size", 1.0),
                self._calculate_position_size_limit(market_data)
            )
            
            step_risks = [
                self._calculate_step_risk(step, market_data)
                for step in strategy_steps
            ]
            
            return StrategyRiskMetrics(
                position_size_limit=position_size_limit,
                max_leverage_ratio=protocol_config.get("max_leverage", 2.0),
                min_collateral_ratio=protocol_config.get("min_collateral_ratio", 1.25),
                rebalance_threshold=self._calculate_rebalance_threshold(market_data),
                stop_loss_level=self._calculate_stop_loss(market_data),
                take_profit_level=self._calculate_take_profit(market_data),
                max_drawdown_limit=self._calculate_max_drawdown_limit(market_data),
                concentration_limit=self._calculate_concentration_limit(market_data),
                step_risks=step_risks,
                protocol_exposure_limits=self._calculate_protocol_exposure_limits(market_data)
            )
        except Exception as e:
            logger.error(f"Error calculating strategy risk metrics: {e}")
            raise

    def _calculate_position_size_limit(self, market_data: Dict) -> float:
        liquidity = float(market_data.get("liquidity", 0))
        volume = float(market_data.get("volume_24h", 0))
        return min(0.1, liquidity * 0.01, volume * 0.05)

    def _calculate_step_risk(self, step: Dict, market_data: Dict) -> Dict[str, float]:
        return {
            "execution_risk": self._calculate_execution_risk(step, market_data),
            "slippage_risk": self._calculate_slippage_risk(step, market_data),
            "timing_risk": self._calculate_timing_risk(step, market_data)
        }

    def _calculate_execution_risk(self, step: Dict, market_data: Dict) -> float:
        return min(1.0, step.get("size", 0) / market_data.get("liquidity", 1))

    def _calculate_slippage_risk(self, step: Dict, market_data: Dict) -> float:
        return min(1.0, step.get("size", 0) / market_data.get("volume_24h", 1))

    def _calculate_timing_risk(self, step: Dict, market_data: Dict) -> float:
        volatility = market_data.get("volatility_24h", 0)
        return min(1.0, volatility * 2)

    def _calculate_rebalance_threshold(self, market_data: Dict) -> float:
        volatility = market_data.get("volatility_24h", 0)
        return max(0.02, min(0.1, volatility * 0.5))

    def _calculate_stop_loss(self, market_data: Dict) -> float:
        volatility = market_data.get("volatility_24h", 0)
        return max(0.05, min(0.2, volatility * 2))

    def _calculate_take_profit(self, market_data: Dict) -> float:
        stop_loss = self._calculate_stop_loss(market_data)
        return stop_loss * 1.5

    def _calculate_max_drawdown_limit(self, market_data: Dict) -> float:
        volatility = market_data.get("volatility_24h", 0)
        return max(0.1, min(0.3, volatility * 3))

    def _calculate_concentration_limit(self, market_data: Dict) -> float:
        return min(0.2, market_data.get("liquidity", 0) / market_data.get("tvl", 1))

    def _calculate_protocol_exposure_limits(self, market_data: Dict) -> Dict[str, float]:
        tvl = market_data.get("tvl", 0)
        return {
            "total": min(0.1, 1000000 / tvl) if tvl > 0 else 0.1,
            "per_asset": min(0.05, 500000 / tvl) if tvl > 0 else 0.05
        }

    async def analyze_risk(
        self,
        asset_address: str,
        protocol_id: str,
        market_data: Optional[Dict] = None,
        strategy_steps: Optional[List[Dict]] = None
    ) -> ComprehensiveRiskMetrics:
        try:
            if market_data is None:
                market_data = await self.market_service.get_latest_market_data()
            
            if strategy_steps is None:
                strategy_steps = []
            
            volatility = await self.calculate_volatility_metrics(market_data.get("price_history", []))
            liquidity = await self.calculate_liquidity_metrics(market_data)
            protocol = await self.calculate_protocol_metrics(market_data)
            market = await self.calculate_market_metrics(market_data)
            correlation = await self.calculate_correlation_metrics(
                market_data.get("asset_returns", []),
                market_data.get("market_returns", []),
                market_data.get("other_assets_returns", {})
            )
            smart_contract = await self.calculate_smart_contract_metrics(market_data)
            collateral = await self.calculate_collateral_metrics(market_data)
            impermanent_loss = await self.calculate_impermanent_loss(market_data)
            strategy_risk = await self.calculate_strategy_risk(protocol_id, market_data, strategy_steps)
            
            risk_scores = {}
            for category in RiskCategory:
                metrics = self._extract_category_metrics(category, market_data)
                weights = self._get_category_weights(category)
                risk_scores[category] = self.calculate_risk_score(category, metrics, weights)
            
            overall_metrics = {
                category.value: score.score 
                for category, score in risk_scores.items()
            }
            
            overall_risk_score = self.calculate_risk_score(
                RiskCategory.MARKET,
                overall_metrics,
                self.risk_weights
            )
            
            return ComprehensiveRiskMetrics(
                asset_address=asset_address,
                protocol_id=protocol_id,
                timestamp=datetime.utcnow(),
                volatility=volatility,
                liquidity=liquidity,
                protocol=protocol,
                market=market,
                correlation=correlation,
                smart_contract=smart_contract,
                collateral=collateral,
                impermanent_loss=impermanent_loss,
                strategy_risk=strategy_risk,
                risk_scores=risk_scores,
                overall_risk_score=overall_risk_score,
                historical_scores=[],
                trend_indicators=self._calculate_trend_indicators(volatility, market_data),
                last_update=datetime.utcnow(),
                data_quality_score=self._calculate_data_quality(market_data),
                warning_flags=self._generate_warning_flags(risk_scores)
            )
        
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            raise

    def _extract_category_metrics(self, category: RiskCategory, data: Dict) -> Dict[str, float]:
        metrics = {}
        if category == RiskCategory.MARKET:
            metrics = {
                "price_volatility": data.get("price_volatility", 0),
                "market_depth": data.get("market_depth", 0),
                "volume_profile": data.get("volume_profile", 0)
            }
        elif category == RiskCategory.SMART_CONTRACT:
            metrics = {
                "audit_score": data.get("audit_score", 0),
                "vulnerability_score": data.get("vulnerability_score", 0),
                "complexity_score": data.get("complexity_score", 0)
            }
        return metrics

    def _get_category_weights(self, category: RiskCategory) -> Dict[str, float]:
        weights = {
            "default_weight": 1.0
        }
        return weights

    def _calculate_trend_indicators(self, volatility: VolatilityMetrics, market_data: Dict) -> Dict[str, float]:
        return {
            "volatility_trend": volatility.volatility_trend,
            "price_momentum": float(market_data.get("price_momentum", 0)),
            "volume_trend": float(market_data.get("volume_trend", 0))
        }

    def _calculate_data_quality(self, market_data: Dict) -> float:
        required_fields = ["price_history", "volume", "tvl"]
        available_fields = sum(1 for field in required_fields if market_data.get(field))
        return available_fields / len(required_fields)

    def _generate_warning_flags(self, risk_scores: Dict[RiskCategory, RiskScore]) -> List[str]:
        warnings = []
        for category, score in risk_scores.items():
            if score.score > score.warning_threshold:
                warnings.append(f"High {category.value} risk detected: {score.score:.2f}")
            if score.score > score.critical_threshold:
                warnings.append(f"CRITICAL: {category.value} risk exceeds safe threshold")
        return warnings

    async def save_risk_metrics(self, metrics: ComprehensiveRiskMetrics) -> None:
        """Save risk metrics to vector store."""
        try:
            key = f"risk_metrics_{metrics.asset_address}_{int(metrics.timestamp.timestamp())}"
            
            metrics_dict = asdict(metrics)
            
            metadata = {
                "type": "risk_metrics",
                "asset_address": metrics.asset_address,
                "protocol_id": metrics.protocol_id,
                "timestamp": metrics.timestamp.isoformat(),
                "overall_risk_score": metrics.overall_risk_score.score,
                "risk_level": metrics.overall_risk_score.level.name,
                "data": metrics_dict
            }
            
            text = f"Risk analysis for {metrics.asset_address} on {metrics.protocol_id} at {metrics.timestamp}"
            
            await self.vector_service.save(key=key, metadata=metadata, text=text)
            logger.info(f"Saved risk metrics with key: {key}")
            
        except Exception as e:
            logger.error(f"Error saving risk metrics: {e}")
            raise 
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class RiskCategory(Enum):
    MARKET = "market"
    LIQUIDITY = "liquidity"
    PROTOCOL = "protocol"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    IMPERMANENT_LOSS = "impermanent_loss"
    SMART_CONTRACT = "smart_contract"
    COLLATERAL = "collateral"

@dataclass
class CorrelationMetrics:
    correlation_matrix: Dict[str, Dict[str, float]]
    beta: float
    systematic_risk: float
    idiosyncratic_risk: float
    tail_dependency: float
    contagion_risk: float

@dataclass
class ImpermanentLossMetrics:
    il_ratio: float
    price_impact: float
    projected_il: float
    historical_il: float
    pool_composition: Dict[str, float]
    token_weights: Dict[str, float]
    rebalance_frequency: int
    optimal_ranges: Dict[str, Dict[str, float]]
    divergence_loss: float
    hedge_effectiveness: float

@dataclass
class VolatilityMetrics:
    daily_volatility: float
    weekly_volatility: float
    monthly_volatility: float
    annual_volatility: float
    volatility_trend: float
    max_drawdown: float
    volatility_spikes: int

@dataclass
class LiquidityMetrics:
    total_tvl: float
    daily_volume: float
    liquidity_depth: float
    slippage_impact: float
    liquidity_concentration: float
    withdrawal_limits: float
    liquidity_utilization: float
    pool_imbalance: float

@dataclass
class ProtocolMetrics:
    tvl_change_24h: float
    unique_users_24h: int
    health_factor: float
    utilization_rate: float
    total_value_locked: float
    hack_history: List[str]
    audit_status: bool
    governance_score: float
    insurance_coverage: float
    timelock_period: int
    admin_keys: int
    upgradeable: bool
    bug_bounty_size: float

@dataclass
class SmartContractMetrics:
    audit_count: int
    last_audit_date: datetime
    critical_vulnerabilities_found: int
    critical_vulnerabilities_fixed: int
    code_coverage: float
    external_dependencies: int
    complexity_score: float
    permission_complexity: float
    
@dataclass
class StrategyRiskMetrics:
    position_size_limit: float
    max_leverage_ratio: float
    min_collateral_ratio: float
    rebalance_threshold: float
    stop_loss_level: float
    take_profit_level: float
    max_drawdown_limit: float
    concentration_limit: float
    step_risks: List[Dict[str, float]]
    protocol_exposure_limits: Dict[str, float]
    
@dataclass
class CollateralMetrics:
    collateral_ratio: float
    liquidation_threshold: float
    liquidation_penalty: float
    max_leverage: float
    required_collateral_types: List[str]
    collateral_volatility: float
    cross_collateral_risk: float
    oracle_dependency: int

@dataclass
class MarketMetrics:
    price: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    fully_diluted_valuation: float
    mcap_tvl_ratio: float
    volume_tvl_ratio: float

@dataclass
class RiskScore:
    category: RiskCategory
    score: float
    level: RiskLevel
    confidence: float
    timestamp: datetime
    factors: Dict[str, float]
    recommendations: List[str]
    warning_threshold: float
    critical_threshold: float

@dataclass
class ComprehensiveRiskMetrics:
    asset_address: str
    protocol_id: str
    timestamp: datetime
    volatility: VolatilityMetrics
    liquidity: LiquidityMetrics
    protocol: ProtocolMetrics
    market: MarketMetrics
    correlation: CorrelationMetrics
    smart_contract: SmartContractMetrics
    collateral: CollateralMetrics
    impermanent_loss: ImpermanentLossMetrics
    strategy_risk: StrategyRiskMetrics
    risk_scores: Dict[RiskCategory, RiskScore]
    overall_risk_score: RiskScore
    historical_scores: List[RiskScore]
    trend_indicators: Dict[str, float]
    last_update: datetime
    data_quality_score: float
    warning_flags: List[str]

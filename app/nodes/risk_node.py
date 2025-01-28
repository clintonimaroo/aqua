import logging
from typing import Dict
from app.risk.analyzer import RiskAnalyzer
from app.risk.types import ComprehensiveRiskMetrics
from langchain_core.messages import AIMessage

logger = logging.getLogger("risk_node")

async def assess_risk(state: dict) -> dict:

    try:
        risk_analyzer = RiskAnalyzer()
        
        asset = state.get("asset")
        protocol = state.get("protocol")
        market_data = state.get("market_analysis", {})
        
        if not asset or not protocol:
            raise ValueError("Missing required state variables: asset and protocol")
        
        risk_metrics: ComprehensiveRiskMetrics = await risk_analyzer.analyze_risk(
            asset_address=asset,
            protocol_id=protocol,
            market_data=market_data
        )
        
        await risk_analyzer.save_risk_metrics(risk_metrics)
        
        risk_message = format_risk_message(risk_metrics)
        
        state["messages"] = state.get("messages", []) + [AIMessage(content=risk_message)]
        state["risk_analysis"] = {
            "metrics": risk_metrics,
            "overall_score": risk_metrics.overall_risk_score.score,
            "risk_level": risk_metrics.overall_risk_score.level.name,
            "recommendations": risk_metrics.overall_risk_score.recommendations
        }
        
        return state
    
    except Exception as e:
        logger.error(f"Error assessing risk: {e}", exc_info=True)
        error_message = f"Risk analysis failed: {str(e)}"
        state["messages"] = state.get("messages", []) + [AIMessage(content=error_message)]
        return state

def format_risk_message(risk_metrics: ComprehensiveRiskMetrics) -> str:
    """Format risk metrics into a human-readable message."""
    
    message = [
        f"Risk Analysis Report",
        f"Overall Risk Level: {risk_metrics.overall_risk_score.level.name}",
        f"Risk Score: {risk_metrics.overall_risk_score.score:.2f}/100",
        f"Confidence: {risk_metrics.overall_risk_score.confidence:.2%}\n"
    ]
    
    message.extend([
        "Key Metrics:",
        f"- Volatility (24h): {risk_metrics.volatility.daily_volatility:.2%}",
        f"- TVL: ${risk_metrics.liquidity.total_tvl:,.2f}",
        f"- Health Factor: {risk_metrics.protocol.health_factor:.2f}",
        f"- Market Beta: {risk_metrics.correlation.beta:.2f}\n"
    ])
    
    message.append("Risk Breakdown:")
    for category, score in risk_metrics.risk_scores.items():
        message.append(f"- {category.value}: {score.score:.2f}/100 ({score.level.name})")
    
    if risk_metrics.overall_risk_score.recommendations:
        message.extend([
            "\nRecommendations:",
            *[f"- {rec}" for rec in risk_metrics.overall_risk_score.recommendations]
        ])
    
    if risk_metrics.warning_flags:
        message.extend([
            "\nWarnings:",
            *[f"- {warning}" for warning in risk_metrics.warning_flags]
        ])
    
    return "\n".join(message)



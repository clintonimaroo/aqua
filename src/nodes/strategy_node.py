import logging
from src.config.llm_config import GPT3_5_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts.strategy_prompts import StrategyGenerationPrompt
from src.config.min_deposits import MIN_DEPOSITS
from src.config.protocols_config import PROTOCOLS_CONFIG
from src.config.strategy_config import STRATEGY_CONFIG
from src.config.addresses_config import CONNECTORS, BASE_TOKENS

logger = logging.getLogger("strategy_node")

async def generate_strategy(state: dict) -> dict:
    try:
        messages = state.get("messages", [])
        risk_analysis = state.get("risk_analysis", "")
        asset = state.get("asset")
        protocol = state.get("protocol")

        protocol_config = PROTOCOLS_CONFIG[protocol]
        strategy_config = STRATEGY_CONFIG.get(asset.lower(), {})
        base_token_address = BASE_TOKENS[asset]
        min_deposit = MIN_DEPOSITS.get(asset, "0")

        markets = protocol_config["addresses"]["markets"]
        model = ChatOpenAI(model=GPT3_5_MODEL)

        messages = [
            SystemMessage(content=StrategyGenerationPrompt.get_system_prompt()),
            HumanMessage(content=StrategyGenerationPrompt.get_human_prompt(
                risk_analysis=risk_analysis,
                asset=asset,
                protocol=protocol,
                protocol_config=protocol_config,
                strategy_config=strategy_config,
                base_token_address=base_token_address,
                min_deposit=min_deposit,
                market_addresses=markets,
                connector=CONNECTORS[protocol]
            ))
        ]
        strategy_response = await model.ainvoke(messages)
       
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"Generated DeFi strategy: {strategy_response}")
        ]
        state["strategy_signals"] = strategy_response
        return state
    
    except Exception as e:
        logger.error(f"Error generating DeFi strategy: {e}", exc_info=True)
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="Error generating DeFi strategy. Please try again later.")
        ]
        return state

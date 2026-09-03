from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def check_langsmith_tracing() -> None:
    """Log whether LangSmith tracing is enabled for this run."""
    tracing_on = config.LANGSMITH_TRACING.lower() == "true"

    if tracing_on and config.LANGSMITH_API_KEY:
        logger.info(
            "LangSmith tracing ENABLED - project '%s', traces at %s",
            config.LANGSMITH_PROJECT,
            "https://smith.langchain.com",
        )
    else:
        logger.info("LangSmith tracing is OFF (set LANGSMITH_TRACING=true in .env to enable)")
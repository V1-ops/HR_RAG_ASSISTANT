from langchain.agents import create_agent


from hr_assistant import config


def create_hr_agent(llm,tools):
    """Return a Langchain agent that can call out tools to answer questions"""
    return create_agent(model = llm, tools= tools,system_prompt= config.SYSTEM_PROMPT)

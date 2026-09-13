from langchain_groq import ChatGroq
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT


from hr_assistant import config
from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask, build_hr_assistant
from hr_assistant.vector_store import get_retriever, load_vector_store

logger =get_logger(__name__)

DATASET_NAME = "hr_policy_qna"


TEST_CASES = [
    {
        "question": "How many paid sick days does an employee receive per calendar year?",
        "answer": "10 paid sick days"
    },
    {
        "question": "How many weeks of fully paid parental leave does a primary caregiver receive?",
        "answer": "16 weeks"
    },
    {
        "question": "How many weeks of fully paid parental leave does a secondary caregiver receive?",
        "answer": "4 weeks"
    },
    {
        "question": "How many days of unused PTO can be carried over into the following calendar year?",
        "answer": "Up to 5 days"
    },
    {
        "question": "How many days in advance must casual or vacation leave exceeding 3 consecutive days be requested?",
        "answer": "At least 10 business days in advance"
    },
    {
        "question": "What are the core operational hours for remote employees?",
        "answer": "10:00 AM to 4:00 PM EST"
    },
    {
        "question": "How much is the one-time home office setup allowance for remote-designated employees?",
        "answer": "$500"
    },
    {
        "question": "How much is the monthly internet reimbursement for remote-designated employees?",
        "answer": "$60 per month"
    },
    {
        "question": "How long does HR have to initiate an investigation after receiving a harassment complaint?",
        "answer": "Within 48 hours"
    },
    {
        "question": "What is the standard duration of a Performance Improvement Plan?",
        "answer": "60 calendar days"
    }
]

def _get_judge_llm() -> ChatGroq:
    """Return the judge model using the same Groq setup as the main app."""
    return ChatGroq(model_name=config.LLM_MODEL_NAME, temperature=0)
    
# if dataset is there reuse it , if not create a new dataset 
# question paper 
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", 
        DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
            "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset

# start the exam
def run_evaluation():
    """Upload the dataset (if needed) 
    and run the correctness evaluation."""
    client = Client()
    dataset = _ensure_dataset(client)

    # Built once and reused for every test case, instead of rebuilding
    # the whole agent (and reconnecting to Qdrant) 10 times over.
    agent = build_hr_assistant()
    retriever = get_retriever(load_vector_store())

    # write the answers 
    def target(inputs: dict) -> dict:
        """
        Run one test question through the real agent, 
        and also capture
        the retrieved chunks 
        so groundedness can check the answer against
        what was actually retrieved 
        (not just the reference answer).
        """
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
        return {"answer": answer, "context": context}

    # giving marks 
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=_get_judge_llm(),
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=_get_judge_llm(),
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check the answer is supported by the retrieved context, not invented."""
        return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"])

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator,
                groundedness_evaluator],
        experiment_prefix="hr-policy-evalzz",
        description="HR policy assistant correctness + groundedness evaluation",
    )

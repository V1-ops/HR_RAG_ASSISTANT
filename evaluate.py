from hr_assistant.evaluation import run_evaluation


def main():
    print("Running HR policy assistant evaluation...")
    results = run_evaluation()
    print("Done. Open your LangSmith project to see the experiment.")
    print(results)


if __name__ == "__main__":
    main()
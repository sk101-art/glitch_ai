from ollama_client import generate_sync, build_prompt, SYSTEM_COMPLIANCE

def main():
    question = "Summarize KYC periodicity for low-risk customers."
    prompt = build_prompt(SYSTEM_COMPLIANCE, question)
    print("=== Model Output ===")
    print(generate_sync(prompt))

if __name__ == "__main__":
    main()

import argparse
import json
from tasks import LLMTask, EmbeddingTask, AudioTask, VisionTask


def main():
    parser = argparse.ArgumentParser(description="Test runner for FLM models.")
    parser.add_argument('--llm', action='store_true', help="Run LLM tests")
    parser.add_argument('--embedding', action='store_true', help="Run Embedding tests")
    parser.add_argument('--audio', action='store_true', help="Run Audio tests")
    parser.add_argument('--vision', action='store_true', help="Run vision tests")
    parser.add_argument('--all', action='store_true', help="Run all available tests")
    parser.add_argument('--gen-lim', type=int, default=-1, help="Maximum number of tokens to generate")
    parser.add_argument('--backend-os', type=str, default="linux", choices=["linux", "windows"], help="OS of the FLM backend (default: linux)")

    args = parser.parse_args()

    if args.all:
        args.llm = args.embedding = args.audio = args.vision = True

    if not any([args.llm, args.embedding, args.audio, args.vision]):
        parser.print_help()
        return

    try:
        print("Please ensure you have started the FLM server and have the correct URL and port. \n")

        with open("backend.json", "r") as f:
            backend_config = json.load(f)
            if "base_url" in backend_config:
                baseurl = backend_config["base_url"]
                print(f"Using base URL from backend.json: {baseurl}")
            else:
                print("No base_url found in backend.json.")

        if args.llm:
            LLMTask(baseurl, args.backend_os).run(max_completion_tokens=args.gen_lim)

        if args.embedding:
            EmbeddingTask(baseurl, args.backend_os).run()

        if args.audio:
            AudioTask(baseurl, args.backend_os).run()

        if args.vision:
            VisionTask(baseurl, args.backend_os).run(max_generation_tokens=args.gen_lim)

    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
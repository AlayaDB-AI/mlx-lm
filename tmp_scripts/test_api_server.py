import argparse
import json
import sys
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc


def stream_chat(url: str, payload: dict, timeout: float, *, print_chunks: bool) -> str:
    payload = dict(payload)
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                decoded = line.decode("utf-8").strip()
                if not decoded or decoded.startswith(":"):
                    continue
                if not decoded.startswith("data:"):
                    continue
                data_str = decoded[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or choices[0].get("message") or {}
                piece = delta.get("content") or ""
                if piece:
                    if print_chunks:
                        print(piece, end="", flush=True)
                    content += piece
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    return content


def extract_content(response_text: str) -> str:
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Response is not valid JSON.") from exc
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Response JSON does not match chat completion schema.") from exc


def run_interactive(args) -> int:
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    messages = []
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            messages.append({"role": "user", "content": user})
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
            }
            if args.stream:
                print("AI: ", end="", flush=True)
                reply = stream_chat(
                    url, payload, args.timeout, print_chunks=True
                )
                print("")
            else:
                response_text = post_json(url, payload, args.timeout)
                reply = extract_content(response_text)
                print(f"AI: {reply}")
            messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\nExiting.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick test client for AlayaJet OpenAI-compatible API server."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Base URL for the API server (default: http://127.0.0.1:8080/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default_model",
        help="Model name to send in the request (default: default_model)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Hello from AlayaJet API server!",
        help="User message content to send",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--print-content-only",
        action="store_true",
        help="Print only the assistant content instead of full JSON",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="Run an interactive chat loop (default)",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable interactive mode",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Use streaming responses (SSE) and print tokens incrementally (default)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming responses",
    )
    args = parser.parse_args()

    if args.interactive:
        return run_interactive(args)

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "temperature": args.temperature,
    }

    if args.stream:
        content = stream_chat(url, payload, args.timeout, print_chunks=True)
        if args.print_content_only:
            if not content.endswith("\n"):
                print("")
        else:
            print("")
            print(json.dumps({"content": content}, ensure_ascii=False))
    else:
        response_text = post_json(url, payload, args.timeout)
        if args.print_content_only:
            content = extract_content(response_text)
            print(content)
        else:
            print(response_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

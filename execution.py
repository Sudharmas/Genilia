import subprocess
import sys
import os
import time

python_executable = sys.executable

script_dir = os.path.dirname(os.path.abspath(__file__))

commands = {
    "rag_agent": [
        python_executable,
        "-m", "uvicorn",
        "rag_agent.main:app",
        "--port", "8000"
    ],
    "action_agent": [
        python_executable,
        os.path.join(script_dir, "action_agent", "main.py")
    ],
    "finetune_agent": [  # <-- NEW AGENT
        python_executable,
        os.path.join(script_dir, "finetune_agent", "main.py")
    ],
    "mcp_server": [
        python_executable,
        os.path.join(script_dir, "mcp_server", "main.py")
    ]
}


processes = {}
print(f"Starting all 4 backend services using {python_executable}...")
print("-" * 30)

try:
    for name, cmd in commands.items():
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        processes[name] = process
        print(f"Launched {name} (PID: {process.pid})")
        time.sleep(3)

    print("-" * 30)
    print("\nAll backend services are running.")
    print("Your system is live!")
    print("\nNOTE: This script does NOT start the frontend React app.")
    print("You must still run 'npm run dev' in the 'frontend' folder.")
    print("\nPress Ctrl+C in this terminal to stop all 4 backend services.")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n--- Shutting down all services ---")

finally:
    for name, process in processes.items():
        if process.poll() is None:
            print(f"Stopping {name} (PID: {process.pid})...")
            process.terminate()

    print("All backend services stopped.")
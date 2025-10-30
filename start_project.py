import os
import sys
import subprocess
from pathlib import Path
import platform
import shutil


def run_command(command, cwd=None):
    """Run a shell command and print output live."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(1)


def open_new_terminal(command, cwd):
    """Open a new terminal window and run a command (Windows/macOS/Linux).
       - 在 VS Code 集成终端里：就在当前终端面板中启动（可见输出）
       - 在普通终端里：弹出新窗口运行
    """
    sysname = platform.system()
    cwd_str = str(cwd)

    # 如果在 VS Code 集成终端中，直接在当前面板里启动，让输出可见
    if os.environ.get("TERM_PROGRAM", "").lower() == "vscode":
        if sysname == "Windows":
            # 继承当前控制台，在同一面板输出；非阻塞
            subprocess.Popen(command, cwd=cwd_str, shell=True)
        elif sysname == "Darwin":
            subprocess.Popen(f'cd "{cwd_str}" && {command}', cwd=cwd_str, shell=True, executable="/bin/bash")
        else:
            subprocess.Popen(f'cd "{cwd_str}" && {command}', cwd=cwd_str, shell=True)
        return

    # 非 VS Code：走“新窗口”逻辑
    if sysname == "Windows":
        # 强制新控制台，不依赖 start；窗口会保留（cmd /k）
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        cmdline = f'cd /d "{cwd_str}" && {command}'
        subprocess.Popen(["cmd.exe", "/k", cmdline], creationflags=creationflags)
    elif sysname == "Darwin":
        # macOS：AppleScript 打开 Terminal
        full_cmd = f'cd "{cwd_str}" && {command}'
        apple_script = f'''
        tell application "Terminal"
            do script "{full_cmd.replace('"', '\\"')}"
            activate
        end tell
        '''
        subprocess.run(["osascript", "-e", apple_script])
    else:
        # Linux：尝试常见终端；找不到则退回当前窗口后台运行
        term = (
            shutil.which("x-terminal-emulator")
            or shutil.which("gnome-terminal")
            or shutil.which("konsole")
            or shutil.which("xfce4-terminal")
            or shutil.which("xterm")
        )
        if term and "gnome-terminal" in term:
            subprocess.Popen([term, "--", "bash", "-lc", f'cd "{cwd_str}" && {command}; exec bash'])
        elif term and "konsole" in term:
            subprocess.Popen([term, "-e", "bash", "-lc", f'cd "{cwd_str}" && {command}; exec bash'])
        elif term and "xfce4-terminal" in term:
            subprocess.Popen([term, "--hold", "-e", f'bash -lc \'cd "{cwd_str}" && {command}; exec bash\''])
        elif term and "xterm" in term:
            subprocess.Popen([term, "-hold", "-e", f'bash -lc \'cd "{cwd_str}" && {command}; exec bash\''])
        elif term:
            subprocess.Popen([term, "-e", f'bash -lc \'cd "{cwd_str}" && {command}; exec bash\''])
        else:
            # 无 GUI 终端可用：在当前窗口后台跑
            subprocess.Popen(f'cd "{cwd_str}" && {command}', cwd=cwd_str, shell=True)


def main():
    root = Path(__file__).parent.resolve()
    frontend_dir = root / "frontend"
    backend_dir = root / "backend"

    print("=== Dynamic Field Boundary Detection: Project Launcher ===")

    # 1. Check Node.js and npm
    print("\n[1/6] Checking Node.js and npm ...")
    try:
        subprocess.run("node -v", shell=True, check=True, stdout=subprocess.PIPE)
        subprocess.run("npm -v", shell=True, check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Node.js or npm not found. Please install them first.")
        sys.exit(1)

    # 2. Install frontend dependencies
    print("\n[2/6] Installing frontend dependencies ...")
    if (frontend_dir / "package.json").exists():
        run_command("npm install", cwd=frontend_dir)
    else:
        print("No package.json found in frontend/. Skipping frontend installation.")

    # 3. Check Python and pip
    print("\n[3/6] Checking Python environment ...")
    if sys.version_info < (3, 9):
        print("Warning: Python 3.9+ is recommended.")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("pip not found. Please install pip first.")
        sys.exit(1)

    # 4. Install backend dependencies
    print("\n[4/6] Installing backend dependencies ...")
    backend_requirements = backend_dir / "requirements.txt"
    if backend_requirements.exists():
        run_command(f"{sys.executable} -m pip install -r {backend_requirements}", cwd=backend_dir)
    else:
        print("No backend/requirements.txt found. Skipping backend installation.")

    # 5. Start backend and frontend in new terminals / current VS Code终端
    print("\n[5/6] Starting backend and frontend servers ...")

    # ⚠️ 给 Python 路径加引号，防止将来路径含空格
    backend_cmd = f"\"{sys.executable}\" -m uvicorn app:app --reload --port 5000"
    frontend_cmd = "npm run dev"

    print("Launching backend in a new terminal ...")
    open_new_terminal(backend_cmd, backend_dir)

    print("Launching frontend in a new terminal ...")
    open_new_terminal(frontend_cmd, frontend_dir)

    print("\n[6/6] All setup complete.")
    print("----------------------------------------------------")
    print("Backend:  http://127.0.0.1:5000")
    print("Frontend: http://localhost:5173")
    print("----------------------------------------------------")
    print("Both servers are running in separate terminal windows.")
    print("Press Ctrl+C here to exit this launcher (servers stay alive).")


if __name__ == "__main__":
    main()



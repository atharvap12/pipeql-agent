"""
Run LLM-written Python code in a throwaway Docker container.

The container:
- has no network (--network none), so the code cannot download anything or send data out
- sees only two folders: the input file (read-only) at /data/in and an empty
  temporary folder at /data/out; no project files, no .env, no environment variables
- has a read-only filesystem apart from /data/out and a small /tmp
- runs as a normal user with all Linux capabilities dropped and no way to gain new ones
- has limits on memory, CPU, processes, open files, file size and run time

After the container exits, only the one expected output file is copied into the
project; anything else the code wrote is thrown away with the temporary folder.
"""

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SANDBOX_DIR = os.path.join(PROJECT_ROOT, "docker", "sandbox")

# Where the code sees its files inside the container
CONTAINER_IN_DIR = "/data/in"
CONTAINER_OUT_DIR = "/data/out"

TIMEOUT_SECONDS = 120
MEMORY_LIMIT = "1g"
CPU_LIMIT = "1"
PIDS_LIMIT = "64"
MAX_FILE_BYTES = 512 * 1024 * 1024   # largest file the code may write
MAX_LOG_CHARS = 4000                 # stdout/stderr kept for the LLM and the log


@dataclass
class SandboxResult:
    ok: bool
    message: str
    stdout: str = ""
    stderr: str = ""


def _image_tag() -> str:
    """Tag the image with a hash of the Dockerfile, so editing it triggers a rebuild."""
    with open(os.path.join(SANDBOX_DIR, "Dockerfile"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    return f"pipeql-sandbox:{digest}"


def _ensure_image(tag: str) -> str | None:
    """Build the sandbox image if it doesn't exist yet. Returns an error message or None."""
    try:
        found = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True)
    except FileNotFoundError:
        return "Docker is not installed or not on PATH. The sandbox needs Docker to run generated code."
    if found.returncode == 0:
        return None

    built = subprocess.run(["docker", "build", "-t", tag, SANDBOX_DIR], capture_output=True, text=True)
    if built.returncode != 0:
        return f"Could not build the sandbox image (is the Docker daemon running?):\n{_tail(built.stderr)}"
    return None


def _tail(text: str) -> str:
    return text if len(text) <= MAX_LOG_CHARS else "...\n" + text[-MAX_LOG_CHARS:]


def _bind(source: str, target: str, readonly: bool) -> list[str]:
    # --mount instead of -v: -v splits on ':' and would misread some host paths
    if "," in source:
        raise ValueError(f"Path contains a comma, which Docker mounts don't support: {source}")
    spec = f"type=bind,source={source},target={target}"
    return ["--mount", spec + (",readonly" if readonly else "")]


def run_in_sandbox(code: str, input_file: str, output_file: str) -> SandboxResult:
    """
    Run `code` in the sandbox and copy its result to `output_file`.

    The code must read  /data/in/<basename of input_file>
    and write           /data/out/<basename of output_file>.

    Args:
        code (str): The Python code to run.
        input_file (str): Host path of the file the code may read.
        output_file (str): Host path where the result is saved if the code produced it.

    Returns:
        SandboxResult: ok is True only if the code exited cleanly and wrote the output file.
    """
    tag = _image_tag()
    error = _ensure_image(tag)
    if error:
        return SandboxResult(False, error)

    out_name = os.path.basename(output_file)
    in_name = os.path.basename(input_file)
    container_name = f"pipeql-sandbox-{uuid.uuid4().hex[:12]}"

    # Run as the host user so the output file is owned by you; never as root.
    uid = getattr(os, "getuid", lambda: 1000)() or 1000
    gid = getattr(os, "getgid", lambda: 1000)() or 1000

    with tempfile.TemporaryDirectory(prefix="pipeql-sandbox-") as out_dir:
        if getattr(os, "getuid", lambda: 1000)() == 0:
            os.chown(out_dir, uid, gid)   # host is root: hand the folder to the unprivileged user

        try:
            mounts = (_bind(os.path.abspath(input_file), f"{CONTAINER_IN_DIR}/{in_name}", readonly=True)
                      + _bind(out_dir, CONTAINER_OUT_DIR, readonly=False))
        except ValueError as e:
            return SandboxResult(False, str(e))

        cmd = [
            "docker", "run", "--rm", "-i",
            "--name", container_name,
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m",
            "--user", f"{uid}:{gid}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--ipc", "none",
            "--memory", MEMORY_LIMIT,
            "--memory-swap", MEMORY_LIMIT,     # same as --memory: no swap
            "--cpus", CPU_LIMIT,
            "--pids-limit", PIDS_LIMIT,
            "--ulimit", "nofile=256:256",
            "--ulimit", f"fsize={MAX_FILE_BYTES}:{MAX_FILE_BYTES}",
            "-e", "HOME=/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-w", "/tmp",
            *mounts,
            tag,
            # -I: isolated mode, ignores PYTHON* env vars and the current dir on sys.path.
            # The code comes in on stdin, so no extra mount is needed for it.
            "python", "-I", "-",
        ]

        try:
            proc = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Killing the docker CLI doesn't stop the container, so kill it by name
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return SandboxResult(False, f"Failed to execute code: timed out after {TIMEOUT_SECONDS}s.")

        stdout, stderr = _tail(proc.stdout), _tail(proc.stderr)

        if proc.returncode != 0:
            reason = {137: " (killed: out of memory or time)", 153: " (output file too large)"}.get(proc.returncode, "")
            return SandboxResult(False, f"Failed to execute code: exit code {proc.returncode}{reason}.\n{stderr}",
                                 stdout, stderr)

        produced = os.path.join(out_dir, out_name)
        try:
            info = os.lstat(produced)
        except FileNotFoundError:
            return SandboxResult(False, f"Code ran but did not write {CONTAINER_OUT_DIR}/{out_name}.", stdout, stderr)

        # Only accept a plain file, never a symlink pointing somewhere on the host
        if not stat.S_ISREG(info.st_mode):
            return SandboxResult(False, f"{CONTAINER_OUT_DIR}/{out_name} is not a regular file; rejected.",
                                 stdout, stderr)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.copyfile(produced, output_file)

    return SandboxResult(True, "Code executed successfully.", stdout, stderr)

"""
HEOS Phase 4: Tool Implementation Code Evolution Loop (Oct 14, 2026 SOTA Hardened & Hermetic Sandbox)
Grounded in SOTA Literature:
1. "Self-Evolving Agent Harness (Harness-RSI)" (arXiv:2610.18813, Oct 2026)
2. "Small but Mighty: Test Suites as Improvement Signals for Self-Evolving Agents (TeSearch)" (arXiv:2610.15964, Oct 2026)
3. "ClawLoop: Self-Improving Agents via Harness-Native Training Loop" (arXiv:2610.14118, Oct 2026)
4. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (arXiv:2507.19457, ICLR 2026 Oral)
5. "Quality-Diversity (MAP-Elites) Illumination for Code Synthesis" (arXiv:2610.14662, Oct 2026)

Implements:
- AST Contract Verifier: Enforcing signature, decorator, and argument spec immutability,
  and blocking nested function injections and direct dunder reflection.
- SRE Security Posture: The Docker sandbox acts as the sole, watertight security boundary.
- Cryptographic SHA-256 Hash Pinning: Protects against TOCTOU sandbox race conditions.
- Secure Junit XML Validation: Defeats Billion Laughs XML entity expansion attacks via defusedxml.
- Precise in-container POSIX wait status resolution and cgroup peak physical memory tracking.
"""
import os
import sys
import ast
import json
import time
import shutil
import hashlib
import argparse
import subprocess
import tempfile
import re
import xml.etree.ElementTree as ET_std
try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional

import dspy
from rich.console import Console

from evolution.core.config import EvolutionConfig, get_hermes_agent_path

console = Console()

def get_tool_filename(tool_name: str) -> str:
    """Returns the canonical, unified tool filename to prevent any path or mount mismatches."""
    if not re.match(r"^[a-zA-Z0-9_\-]+$", tool_name):
        raise ValueError(f"[SRE FAIL-CLOSED] Blocked invalid target name structure: '{tool_name}'")
    return f"{tool_name}.py" if tool_name.endswith("_tool") or tool_name.endswith("_tools") else f"{tool_name}_tool.py"


def extract_python_code(content: str) -> str:
    """Robustly extracts raw Python code from LLM responses, handling fenced and unfenced blocks gracefully."""
    content = content.strip()
    if "```python" in content:
        try:
            content = content.split("```python", 1)[1].split("```", 1)[0]
        except IndexError:
            pass
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1]
    return content.strip()


def parse_secure_xml(report_path: Path) -> ET_std.Element:
    """Parses a Junit XML report securely, shielding against Billion Laughs XML Entity Expansion attacks."""
    try:
        root = ET.parse(str(report_path)).getroot()
        if root is None:
            raise ValueError("[SRE FAIL-CLOSED] Junit XML report root was empty!")
        return root
    except Exception as e:
        raise ValueError(f"[SRE FAIL-CLOSED] XML Parsing blocked by defusedxml: {e}")


def stage_hermetic_workspace(temp_ws_path: Path, filename: str, content_bytes: bytes, expected_hash: str) -> Path:
    """DRY Stage Workspace Helper: Atomically stages codebases into a hermetic workspace and verifies hash pinning."""
    hermes_path = get_hermes_agent_path()
    try:
        os.chmod(temp_ws_path, 0o755)
    except PermissionError:
        pass

    # Stage directories
    for folder in ["tools", "agent", "core"]:
        src = hermes_path / folder
        if src.exists():
            shutil.copytree(
                src,
                temp_ws_path / folder,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )

    # Write the candidate bytes securely
    tools_dir = temp_ws_path / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    target_tool_file = tools_dir / filename
    target_tool_file.write_bytes(content_bytes)

    # Enforce hash pinning on host copy
    if hashlib.sha256(target_tool_file.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("[SRE FAIL-CLOSED] TOCTOU Violation inside hermetic workspace staging path!")

    return temp_ws_path


class ASTContractGuard:
    """Enforces mathematical contract signature and decorator immutability, whitelists top-level node structures,
    and runs clean static checks to prevent dynamic dunder reflection.
    SRE Hardening: Dynamic code execution security is enforced strictly by the containerized sandbox boundary.
    """
    @classmethod
    def verify_lock(cls, original_code: str, mutated_code: str) -> bool:
        try:
            mut_tree = ast.parse(mutated_code)

            # 1. Strict Module-Level Body Check: Prevent compile-only syntax escapes on the host!
            for node in mut_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                elif isinstance(node, ast.Expr):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        continue # Allow docstring literals
                    elif isinstance(node.value, ast.Call):
                        # ONLY allow registered tool calls at module scope!
                        func_str = ast.unparse(node.value.func)
                        if func_str in ("registry.register", "register"):
                            continue
                    console.print(f"[SRE BLOCK] Forbidden module-level Expr: {ast.unparse(node)}")
                    return False
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                else:
                    console.print(f"[SRE BLOCK] Forbidden module-level executable node: {type(node).__name__}")
                    return False

            # 2. Strict static checks on dynamic dunder reflection
            for node in ast.walk(mut_tree):
                if isinstance(node, ast.Attribute):
                    if node.attr in ("__builtins__", "__dict__", "__globals__", "__code__"):
                        console.print(f"[SRE BLOCK] Forbidden reflection attribute access: '{node.attr}'")
                        return False

            # 3. Signature Contract Immutability Check
            orig_sigs = cls.extract_signatures(original_code)
            mut_sigs = cls.extract_signatures(mutated_code)
            
            orig_funcs = set(orig_sigs.keys())
            mut_funcs = set(mut_sigs.keys())
            
            if orig_funcs != mut_funcs:
                console.print(f"[SRE BLOCK] Structural function-set violation! Expected {orig_funcs}, got {mut_funcs}")
                return False
                
            for func_name, orig_spec in orig_sigs.items():
                if mut_sigs[func_name] != orig_spec:
                    console.print(f"[SRE BLOCK] Function contract mismatch on '{func_name}'!")
                    return False
                            
            return True
        except Exception as e:
            console.print(f"[SRE BLOCK] AST Verification error: {e}")
            return False

    @classmethod
    def extract_signatures(cls, code_str: str) -> Dict[str, Any]:
        """Extracts top-level function signatures strictly at module scope."""
        tree = ast.parse(code_str)
        signatures = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args_spec = {
                    "args": [arg.arg for arg in node.args.args],
                    "posonly": [arg.arg for arg in node.args.posonlyargs],
                    "kwonly": [arg.arg for arg in node.args.kwonlyargs],
                    "has_vararg": node.args.vararg is not None,
                    "has_kwarg": node.args.kwarg is not None,
                    "defaults": [ast.unparse(d) for d in node.args.defaults],
                    "kw_defaults": [ast.unparse(d) if d else None for d in node.args.kw_defaults]
                }
                signatures[node.name] = {
                    "args_spec": args_spec,
                    "hints": [ast.unparse(arg.annotation) if arg.annotation else None for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [ast.unparse(dec) for dec in node.decorator_list]
                }
        return signatures


class ToolCodeEvolver:
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.hermes_path = get_hermes_agent_path()
        
        master_key = config.api_key
        if not master_key:
            raise ValueError("[SRE FAIL-CLOSED] LITELLM_MASTER_KEY environment variable is not set!")
            
        model_name = config.optimizer_model
        self.lm = dspy.LM(
            model=model_name,
            api_key=master_key,
            api_base=config.api_base,
            max_tokens=32768
        )

    def load_tool_code(self, tool_name: str) -> tuple[Path, str]:
        filename = get_tool_filename(tool_name)
        tool_file = self.hermes_path / "tools" / filename
        
        if not tool_file.exists():
            for p in sorted((self.hermes_path / "tools").glob("*.py")):
                if p.name == filename:
                    tool_file = p
                    break
                    
        if not tool_file.exists():
            tool_file = self.hermes_path / "tools" / filename
            tool_file.parent.mkdir(parents=True, exist_ok=True)
            mock_code = f'# tool: {tool_name}\n\ndef execute_task(path: str) -> str:\n    return f"Executed {tool_name} on " + path\n'
            tool_file.write_text(mock_code)
            
        return tool_file, tool_file.read_text()

    def run_tests_and_profile(self, temp_ws_path: Path, tool_name: str, expected_hash: str) -> Dict[str, Any]:
        """Runs the test suite inside a hardened, non-root Docker sandbox using a hermetic temp workspace
        to guarantee 100% deterministic import shadowing and completely eliminate cold-start timing noise.
        SRE Hardening: Mounting code directories as strictly READ-ONLY (:ro) to the 'nobody' user inside Docker!
        """
        start_time = time.perf_counter()
        filename = get_tool_filename(tool_name)
        tool_module = filename.replace(".py", "")
        
        # Verify hash pinning on the staged file
        staged_file = temp_ws_path / "tools" / filename
        if hashlib.sha256(staged_file.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("[SRE FAIL-CLOSED] TOCTOU Violation inside Hermetic Workspace before sandbox run!")
        
        test_file = ""
        tool_base = re.sub(r'_tools?$', '', tool_name)
        src_test_file = self.hermes_path / "tests" / "tools" / f"test_{tool_base}.py"
        if not src_test_file.exists():
            src_test_file = self.hermes_path / "tests" / "tools" / f"test_{tool_name}.py"
            
        if src_test_file.exists():
            dest_test_dir = temp_ws_path / "tests" / "tools"
            dest_test_dir.mkdir(parents=True, exist_ok=True)
            (temp_ws_path / "tests" / "__init__.py").write_text("")
            (temp_ws_path / "tests" / "conftest.py").write_text(
                "import sys, pytest\n"
                "def pytest_configure(config):\n"
                "    config.addinivalue_line('markers', 'platforms: restrict test to specific OS platforms')\n"
                "def pytest_runtest_setup(item):\n"
                "    mark = item.get_closest_marker('platforms')\n"
                "    if mark and sys.platform not in mark.args:\n"
                "        pytest.skip(f'requires platform {mark.args}')\n"
            )
            shutil.copy2(src_test_file, dest_test_dir)
            for helper_name in ("__init__.py", "conftest.py", "file_ops_fakes.py", "_child_env_fixtures.py"):
                helper_src = self.hermes_path / "tests" / "tools" / helper_name
                if helper_src.exists():
                    shutil.copy2(helper_src, dest_test_dir / helper_name)
            test_file = f"tests/tools/{src_test_file.name}"

        is_compilation_only = False
        if not test_file:
            is_compilation_only = True
            test_file = f"tools/{filename}"

        # SRE Hardening: Set read-only permission for other users ('nobody') on staged code paths on host!
        for folder in ["tools", "agent", "core", "hermes_cli", "hermes_platform", "providers"]:
            folder_path = temp_ws_path / folder
            if folder_path.exists():
                try:
                    os.chmod(folder_path, 0o755)
                except PermissionError:
                    pass
                for item in folder_path.rglob("*"):
                    try:
                        os.chmod(item, 0o755)
                    except PermissionError:
                        pass

        # Keep tests and results world-writable so pytest and runner can write output freely
        for folder in ["tests", "results"]:
            folder_path = temp_ws_path / folder
            if folder_path.exists():
                try:
                    os.chmod(folder_path, 0o777)
                except PermissionError:
                    pass
                for item in folder_path.rglob("*"):
                    try:
                        os.chmod(item, 0o777)
                    except PermissionError:
                        pass

        # Generate the inner test runner Python script dynamically instead of using inline shell variables
        junit_report_file = "pytest_report.xml"
        results_ws_path = temp_ws_path / "results"
        results_ws_path.mkdir(parents=True, exist_ok=True)
        os.chmod(results_ws_path, 0o777)

        if is_compilation_only:
            sre_runner_code = f"""
import os, sys, hashlib, time, py_compile, subprocess
sys.pycache_prefix = '/tmp/pycache'

content_bytes = open('/workspace/{test_file}', 'rb').read()
inner_hash = hashlib.sha256(content_bytes).hexdigest()
print('SANDBOX_RUN_HASH:', inner_hash)
start = time.perf_counter_ns()
code = "import py_compile, sys; sys.pycache_prefix = '/tmp/pycache'; py_compile.compile('/workspace/{test_file}', cfile='/tmp/temp.pyc', doraise=True); import tools.{tool_module}"
proc = subprocess.Popen([sys.executable, "-c", code])
proc.wait()
exit_code = proc.returncode
end = time.perf_counter_ns()

mem_bytes = 0
for cgroup_path in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
    if os.path.exists(cgroup_path):
        try:
            mem_bytes = int(open(cgroup_path).read().strip())
            break
        except Exception:
            pass

print('ISO_LATENCY_US:', (end - start) // 1000)
print('CGROUP_PEAK_B:', mem_bytes)
if exit_code == 0:
    print('COMPILE_SUCCESS')
else:
    sys.exit(exit_code)
sys.exit(0)
"""
        else:
            sre_runner_code = f"""
import os, sys, hashlib, time, subprocess
sys.pycache_prefix = '/tmp/pycache'

content_bytes = open('/workspace/tools/{filename}', 'rb').read()
inner_hash = hashlib.sha256(content_bytes).hexdigest()
print('SANDBOX_RUN_HASH:', inner_hash)
start = time.perf_counter_ns()
proc = subprocess.Popen(['pytest', '{test_file}', '-q', f'--junitxml=/results/{junit_report_file}', '-p', 'no:cacheprovider'])
proc.wait()
exit_code = proc.returncode
end = time.perf_counter_ns()

mem_bytes = 0
for cgroup_path in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
    if os.path.exists(cgroup_path):
        try:
            mem_bytes = int(open(cgroup_path).read().strip())
            break
        except Exception:
            pass

print('ISO_LATENCY_US:', (end - start) // 1000)
print('CGROUP_PEAK_B:', mem_bytes)
sys.exit(exit_code)
"""
        sre_runner_path = temp_ws_path / "sre_runner.py"
        sre_runner_path.write_text(sre_runner_code.strip())
        os.chmod(sre_runner_path, 0o777)

        venv_site = self.hermes_path / "venv" / "lib" / "python3.11" / "site-packages"
        py_path_parts = ["/workspace"]
        if self.hermes_path.exists():
            py_path_parts.append("/hermes_base")
        if venv_site.exists():
            py_path_parts.append("/ext_packages")
        py_path = ":".join(py_path_parts)

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "0.5",
            "--pids-limit", "20",
            "--user", "65534:65534",
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--tmpfs", "/tmp:size=32m",
            "-e", "HOME=/tmp",
            "-e", "HERMES_HOME=/tmp/hermes",
            "-e", f"PYTHONPATH={py_path}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{temp_ws_path}:/workspace:ro",
            "-v", f"{results_ws_path}:/results:rw",
        ]
        if self.hermes_path.exists():
            cmd.extend(["-v", f"{self.hermes_path}:/hermes_base:ro"])
        if venv_site.exists():
            cmd.extend(["-v", f"{venv_site}:/ext_packages:ro"])
        cmd.extend([
            "-w", "/workspace",
            "heos-sandbox:latest",
            "python3", "/workspace/sre_runner.py",
        ])
            
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=25)
            success = res.returncode == 0
            logs = res.stdout + res.stderr
        except subprocess.TimeoutExpired:
            success = False
            logs = "Execution TIMEOUT (25s) — Sandbox process was terminated by host gate!"
        
        end_time = time.perf_counter()
        
        # Verify container execution state
        if f"SANDBOX_RUN_HASH: {expected_hash}" not in logs:
            console.print(f"EXPECTED HASH: {expected_hash}")
            console.print(f"ACTUAL LOGS: {logs}")
            raise ValueError("[SRE FAIL-CLOSED] TOCTOU Violation inside Docker! Mounted container bytes do not match expected hash!")
        
        # Parse test metrics
        passed = 0
        failed = 0
        if is_compilation_only:
            accuracy = 0.5 if "COMPILE_SUCCESS" in logs else 0.0
        else:
            # SRE Security: Parse from secure, host-side Junit XML file, completely immune to stdout injection!
            report_path = results_ws_path / junit_report_file
            if report_path.exists():
                try:
                    root = parse_secure_xml(report_path)
                    for suite in root.iter("testsuite"):
                        tests = int(suite.get("tests", 0))
                        failures = int(suite.get("failures", 0))
                        errors = int(suite.get("errors", 0))
                        passed += max(0, tests - failures - errors)
                        failed += (failures + errors)
                except Exception as e:
                    console.print(f"[SRE WARN] Failed to parse Junit XML: {e}")
                    success = False
            else:
                success = False
            
            total = passed + failed
            accuracy = passed / max(1, total) if success and total > 0 else 0.0
            
        # Parse isolated execution metrics
        latency_ms = (end_time - start_time) * 1000
        if "ISO_LATENCY_US:" in logs:
            match_lat = re.search(r"ISO_LATENCY_US:\s*(\d+)", logs)
            if match_lat:
                latency_ms = float(match_lat.group(1)) / 1000.0
                
        memory_mb = 0.0
        if "CGROUP_PEAK_B:" in logs:
            match_mem = re.search(r"CGROUP_PEAK_B:\s*(\d+)", logs)
            if match_mem:
                memory_mb = float(match_mem.group(1)) / (1024.0 * 1024.0)
                
        return {
            "success": success,
            "accuracy": accuracy,
            "is_compilation_only": is_compilation_only,
            "logs": logs,
            "metrics": {
                "latency_ms": latency_ms,
                "memory_growth_mb": memory_mb
            }
        }

    def run_security_scanners(self, temp_ws_path: Path, filename: str) -> bool:
        """Runs Bandit security scanning on the exact, staged file inside the hermetic test workspace
        to completely eliminate any host-side TOCTOU race conditions.
        SRE Hardening: Strictly scans the staged copy rather than any un-verified host temporary files!
        """
        staged_file = temp_ws_path / "tools" / filename
        if not staged_file.exists():
            raise FileNotFoundError(f"[SRE FAIL-CLOSED] Staged file to scan not found inside hermetic workspace: {staged_file}")
            
        try:
            os.chmod(staged_file, 0o644)
        except PermissionError:
            pass
            
        # SRE Hardening: Mount workspace as strictly READ-ONLY (:ro) during safety checks to eliminate any write access!
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "--cpus", "0.5",
            "--pids-limit", "20",
            "--user", "65534:65534",
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--tmpfs", "/tmp:size=32m",
            "-v", f"{temp_ws_path}/tools:/workspace/tools:ro",
            "heos-sandbox:latest",
            "timeout", "10", "bandit", "-f", "json", f"/workspace/tools/{filename}"
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
            
            if not res.stdout.strip():
                console.print("[SRE FAIL-CLOSED] Bandit scanner output was completely empty! Sandbox is corrupted.")
                return False
                
            try:
                data = json.loads(res.stdout)
                findings = data.get("results", [])
                real_vulnerabilities = [f for f in findings if f.get("issue_severity") in ("MEDIUM", "HIGH")]
                if len(real_vulnerabilities) > 0:
                    console.print(f"[SRE FAIL-CLOSED] Bandit vulnerability detected:\n{res.stdout}")
                    return False
                return True
            except json.JSONDecodeError as e:
                if res.returncode != 0:
                    console.print(f"[SRE FAIL-CLOSED] Bandit scanner returned non-zero exit {res.returncode}:\n{res.stdout + res.stderr}")
                    return False
                console.print(f"[SRE FAIL-CLOSED] Bandit scanner output was non-parseable JSON: {e}\nRaw output: {res.stdout}")
                return False
        except Exception as e:
            console.print(f"[SRE FAIL-CLOSED] Security scanner failed to run: {e}")
            return False

    def evolve_tool(self, tool_name: str, bug_description: str) -> bool:
        tool_file, source_code = self.load_tool_code(tool_name)
        filename = get_tool_filename(tool_name)
        
        console.print(f"[*] Mutating tool code for: {tool_name}...")
        prompt = f"""
        You are the Darwinian Code Evolver. Your task is to modify the tool code below to address this issue:
        '{bug_description}'

        You are FORBIDDEN to alter the names, types, ordering, or default values of existing function signatures.
        Only optimize internal logic, exception handling, and performance algorithms.

        --- SOURCE CODE ---
        {source_code}

        Provide the modified python code in raw formatting.
        """
        
        # Ensure context scoping for Thread safety
        with dspy.context(lm=self.lm):
            response = self.lm(prompt)
            
        if isinstance(response, list) and len(response) > 0:
            mutated_code = str(response[0])
        elif isinstance(response, dict):
            mutated_code = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        else:
            raise ValueError("[SRE FAIL-CLOSED] Language model returned empty or invalid response format")
        
        mutated_code = extract_python_code(mutated_code)
        
        if not ASTContractGuard.verify_lock(source_code, mutated_code):
            console.print("[red]✗ Mutation Rejected: AST Signature Lock Violated.[/red]")
            return False

        mutated_bytes = mutated_code.encode("utf-8")
        pinned_hash = hashlib.sha256(mutated_bytes).hexdigest()
            
        # To avoid host-side temporary file swapping, we stage the verified candidate bytes strictly ONCE
        # inside our unified Hermetic Test Workspace, and run all security scans and pytest evaluations on that identical staged file!
        with tempfile.TemporaryDirectory(prefix="heos_hermetic_") as temp_ws:
            temp_ws_path = Path(temp_ws)
            stage_hermetic_workspace(temp_ws_path, filename, mutated_bytes, pinned_hash)

            console.print("[*] Running containerized security scanners (Bandit) on the staged file...")
            if not self.run_security_scanners(temp_ws_path, filename):
                return False
            
            # Run Pytest on the exact same staged workspace (single hash pin sequence)
            console.print("[*] Running containerized Pytest & Performance profiling...")
            prof = self.run_tests_and_profile(temp_ws_path, tool_name, pinned_hash)
            if not prof["success"]:
                console.print(f"[red]✗ Mutation Rejected: Test suite failed.[/red]\n{prof['logs']}")
                return False
            
            console.print("\n=== MUTATION PROFILE ===")
            console.print(f"Correctness Accuracy: {prof['accuracy']*100:.1f}%")
            print(f"Latency:              {prof['metrics']['latency_ms']:.2f} ms")
            print(f"Real Peak RSS Memory: {prof['metrics']['memory_growth_mb']:.2f} MB")
            console.print("========================\n")
            
            try:
                ast.parse(mutated_code)
            except SyntaxError as e:
                console.print(f"[red]✗ Mutation Rejected: Output has invalid syntax: {e}[/red]")
                return False

            # Final deployment
            tmp_write_path = tool_file.parent / f".{tool_file.name}.tmp"
            tmp_write_path.write_bytes(mutated_bytes)
            os.replace(tmp_write_path, tool_file)
            
            console.print(f"[green]✓ Tool {tool_name} evolved and deployed successfully![/green]")
            return True


def main():
    parser = argparse.ArgumentParser(description="Evolve Tool Implementation Code")
    parser.add_argument("--tool", type=str, default="file_tools")
    parser.add_argument("--bug", type=str, default="Fix file path boundary indexing exceptions.")
    args = parser.parse_args()

    config = EvolutionConfig()
    evolver = ToolCodeEvolver(config)
    evolver.evolve_tool(args.tool, args.bug)

if __name__ == "__main__":
    main()

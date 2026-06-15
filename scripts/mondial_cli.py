#!/usr/bin/env python3
"""
Mondial-Xboost CLI — entrypoint de conveniencia.

Uso básico:
    mondial                       # muestra el menú interactivo
    mondial --help                # ayuda rápida de comandos
    mondial entrenar              # entrena el modelo canónico
    mondial predecir --home Brazil --away Morocco
    mondial guia                  # guía detallada con ejemplos

Para ver los parámetros de un comando específico:
    mondial entrenar --help
    mondial predecir --help
    mondial loop --help
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows terminals so box-drawing and accented chars render
# without crashing. Characters unsupported by the active code page become '?'.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

# ANSI colors
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "dim": "\033[2m",
    "blue": "\033[94m",
}


def _color(text: str, color: str) -> str:
    """Return colored text if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{C.get(color, '')}{text}{C['reset']}"
    return text


def _run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    """Run a command inside the project root."""
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(ROOT)
    if env:
        run_env.update(env)
    return subprocess.call(cmd, cwd=ROOT, env=run_env)


def _run_shell(cmd: str, env: dict[str, str] | None = None) -> int:
    """Run a shell command inside the project root."""
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(ROOT)
    if env:
        run_env.update(env)
    return subprocess.call(cmd, cwd=ROOT, env=run_env, shell=True)


def _find_venv_python() -> Path:
    """Return the virtual environment python executable if it exists."""
    candidates = [
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / "venv" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(PYTHON)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_install(_args: argparse.Namespace) -> int:
    """Install project dependencies."""
    python = Path(PYTHON)

    # If no virtual environment is active and no local venv exists, create one.
    if os.getenv("VIRTUAL_ENV") is None:
        local_venv = ROOT / "venv"
        if not local_venv.exists() and not (ROOT / ".venv").exists():
            print(f"{C['cyan']}Creando entorno virtual en {local_venv}...{C['reset']}")
            code = subprocess.call([str(python), "-m", "venv", str(local_venv)], cwd=ROOT)
            if code != 0:
                return code
            python = local_venv / "Scripts" / "python.exe"
            if not python.exists():
                python = local_venv / "bin" / "python"

    requirements = ROOT / "requirements.txt"
    if requirements.exists():
        code = _run([str(python), "-m", "pip", "install", "-r", str(requirements)])
        if code != 0:
            return code
    return _run([str(python), "-m", "pip", "install", "-e", "."])


def cmd_train(args: argparse.Namespace) -> int:
    """Train the canonical model."""
    command = [PYTHON, "scripts/train.py", "--engine", args.engine]
    if args.name:
        command.extend(["--name", args.name])
    if args.min_date:
        command.extend(["--min-date", args.min_date])
    if args.elo_decay is not None:
        command.extend(["--elo-decay", str(args.elo_decay)])
    if args.elo_recent is not None:
        command.extend(["--elo-recent", str(args.elo_recent)])
    return _run(command)


def cmd_train_gpu(args: argparse.Namespace) -> int:
    """Train the canonical model using GPU."""
    command = [PYTHON, "scripts/train.py", "--engine", args.engine]
    if args.name:
        command.extend(["--name", args.name])
    if args.min_date:
        command.extend(["--min-date", args.min_date])
    if args.elo_decay is not None:
        command.extend(["--elo-decay", str(args.elo_decay)])
    if args.elo_recent is not None:
        command.extend(["--elo-recent", str(args.elo_recent)])
    env = {"XGBOOST_DEVICE": "cuda"} if args.engine == "xgboost" else {}
    return _run(command, env=env)


def cmd_predict(args: argparse.Namespace) -> int:
    """Run a prediction."""
    command = [PYTHON, "scripts/predict.py", "--home", args.home, "--away", args.away, "--engine", args.engine]
    if args.model:
        command.extend(["--model", args.model])
    if args.date:
        command.extend(["--date", args.date])
    if args.blend:
        command.append("--blend")
    if args.cold_start_only:
        command.append("--cold-start-only")
    return _run(command)


def cmd_train_cold_start(args: argparse.Namespace) -> int:
    """Train the cold-start fallback model."""
    command = [
        PYTHON, "-c",
        (
            "from predictors.feature_engineering import build_training_dataset; "
            "from predictors.cold_start_model import ColdStartPredictor; "
            f"df = build_training_dataset(min_date='{args.min_date}', elo_decay_halflife_years={args.elo_decay or 'None'}, elo_recent_years={args.elo_recent}); "
            f"m = ColdStartPredictor(cold_threshold={args.cold_threshold}, random_state=2026); "
            "print(m.fit(df)); "
            f"m.save('{args.name}')"
        ),
    ]
    return _run(command)


def cmd_test(_args: argparse.Namespace) -> int:
    """Run Python tests."""
    try:
        __import__("pytest")
    except ImportError:
        print(f"{C['red']}[ERROR] pytest no está instalado.{C['reset']}")
        print(f"{C['yellow']}Instalalo con: pip install -e \".[dev]\"{C['reset']}")
        return 1
    return _run([PYTHON, "-m", "pytest", "tests/"])


def cmd_lint(_args: argparse.Namespace) -> int:
    """Run Python lint."""
    try:
        __import__("ruff")
    except ImportError:
        print(f"{C['red']}[ERROR] ruff no está instalado.{C['reset']}")
        print(f"{C['yellow']}Instalalo con: pip install -e \".[dev]\"{C['reset']}")
        return 1
    return _run([PYTHON, "-m", "ruff", "check", "predictors", "scripts", "backtest", "tests"])


def cmd_gates(_args: argparse.Namespace) -> int:
    """Run verification gates."""
    return _run([PYTHON, "scripts/verify_gates.py"])


def cmd_backtest(_args: argparse.Namespace) -> int:
    """Run the World Cup backtest gate."""
    return _run([PYTHON, "scripts/run_backtest_gate.py"])


def cmd_bridge(_args: argparse.Namespace) -> int:
    """Run the C# <-> Python bridge smoke test."""
    return _run([PYTHON, "scripts/run_bridge_smoke_test.py"])


def cmd_elo(_args: argparse.Namespace) -> int:
    """Compare Elo ratings against World Football Elo."""
    return _run([PYTHON, "scripts/compare_elo_worldfootball.py"])


def cmd_audit(_args: argparse.Namespace) -> int:
    """Run temporal leakage audit."""
    return _run([PYTHON, "scripts/audit_leakage.py"])


def cmd_loop(args: argparse.Namespace) -> int:
    """Run hyperparameter tuning with Optuna."""
    command = [PYTHON, "scripts/loop_engineering.py"]
    if args.trials:
        command.extend(["--trials", str(args.trials)])
    if args.auto:
        command.append("--auto")
    return _run(command)


def cmd_auto_loop(args: argparse.Namespace) -> int:
    """Run automated loop engineering: tune, analyze, strategize, retrain, document."""
    command = [PYTHON, "scripts/auto_loop_engineering.py"]
    if args.trials:
        command.extend(["--trials", str(args.trials)])
    if args.name:
        command.extend(["--name", args.name])
    if args.tune_only:
        command.append("--tune-only")
    if args.backtest:
        command.append("--backtest")
    if args.no_walk_forward:
        command.append("--no-walk-forward")
    return _run(command)


def cmd_data_council(_args: argparse.Namespace) -> int:
    """Run the data council review."""
    return _run([PYTHON, "scripts/run_data_council.py"])


def cmd_dashboard(_args: argparse.Namespace) -> int:
    """Launch the training dashboard."""
    return _run([PYTHON, "scripts/training_dashboard.py"])


def cmd_serve(_args: argparse.Namespace) -> int:
    """Start the FastAPI bridge server."""
    return _run([PYTHON, "-m", "predictors.api"])


def cmd_training_server(args: argparse.Namespace) -> int:
    """Start the training dashboard server."""
    command = [PYTHON, "scripts/training_server.py"]
    if args.port:
        command.extend(["--port", str(args.port)])
    if args.host:
        command.extend(["--host", args.host])
    return _run(command)


def cmd_health(_args: argparse.Namespace) -> int:
    """Check the FastAPI bridge health endpoint."""
    return _run_shell("curl -s http://127.0.0.1:8000/health || echo 'Servidor no disponible'", env={})


def cmd_manifest(_args: argparse.Namespace) -> int:
    """Display the current model manifest."""
    manifest_path = ROOT / "data" / "models" / "model_manifest.json"
    if not manifest_path.exists():
        print(f"No existe {manifest_path}", file=sys.stderr)
        return 1
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))
    return 0


def cmd_clean(_args: argparse.Namespace) -> int:
    """Remove generated cache and test artifacts."""
    patterns = [
        "**/__pycache__",
        "**/*.py[cod]",
        "**/*$py.class",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "data/models/test_*_meta.json",
        "data/models/test_*_outcome.pkl",
        "data/models/test_*_home_goals.pkl",
        "data/models/test_*_away_goals.pkl",
    ]
    removed = 0
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f"Eliminado directorio: {path.relative_to(ROOT)}")
                removed += 1
            elif path.is_file():
                path.unlink()
                print(f"Eliminado archivo: {path.relative_to(ROOT)}")
                removed += 1
    print(f"Limpieza completa. {removed} elementos eliminados.")
    return 0


def cmd_info(_args: argparse.Namespace) -> int:
    """Show environment information."""
    venv_python = _find_venv_python()
    print(f"Project root : {ROOT}")
    print(f"Python       : {PYTHON}")
    print(f"Venv python  : {venv_python}")
    print(f"XGBOOST_DEVICE env: {os.getenv('XGBOOST_DEVICE', 'cpu')}")
    code = _run([PYTHON, "-c", "import xgboost, sklearn, pandas; print(f'xgboost {xgboost.__version__} | sklearn {sklearn.__version__} | pandas {pandas.__version__}')"])
    return code


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Run portability/health checks and report problems."""
    print(f"\n{C['bold']}{C['cyan']}Mondial-Xboost — Doctor de portabilidad{C['reset']}\n")
    issues = 0
    warnings = 0

    # Python version
    py_version = sys.version_info
    print(f"  Python       : {sys.executable}")
    print(f"  Versión      : {sys.version.split()[0]}")
    if py_version < (3, 11):
        print(f"  {C['red']}[ERROR] Se requiere Python >= 3.11 (tenés {py_version.major}.{py_version.minor}).{C['reset']}")
        issues += 1
    else:
        print(f"  {C['green']}[OK] Python >= 3.11{C['reset']}")

    # Virtual environment
    active_venv = os.getenv("VIRTUAL_ENV")
    local_venv = ROOT / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
    if active_venv:
        print(f"  {C['green']}[OK] Entorno virtual activo: {active_venv}{C['reset']}")
    elif local_venv.exists():
        print(f"  {C['green']}[OK] Entorno virtual local encontrado: {local_venv.parent}{C['reset']}")
    else:
        print(f"  {C['yellow']}[WARN] No hay entorno virtual activo ni local. Ejecutá 'mondial instalar'.{C['reset']}")
        warnings += 1

    # Core dependencies
    core_deps = {
        "xgboost": "xgboost",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "requests": "requests",
        "optuna": "optuna",
    }
    print(f"\n{C['bold']}Dependencias core:{C['reset']}")
    for pkg, mod in core_deps.items():
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "desconocida")
            print(f"  {C['green']}[OK]{C['reset']} {pkg:15s} {version}")
        except ImportError as exc:
            print(f"  {C['red']}[ERROR]{C['reset']} {pkg:15s} no importable ({exc})")
            issues += 1

    # Optional dependencies
    optional_deps = {
        "pytest": "pytest",
        "ruff": "ruff",
        "httpx": "httpx",
    }
    print(f"\n{C['bold']}Dependencias opcionales (dev):{C['reset']}")
    for pkg, mod in optional_deps.items():
        try:
            module = __import__(mod)
            version = getattr(module, "__version__", "desconocida")
            print(f"  {C['green']}[OK]{C['reset']} {pkg:15s} {version}")
        except ImportError:
            print(f"  {C['yellow']}[WARN]{C['reset']} {pkg:15s} no instalado")
            warnings += 1

    # Dataset
    print(f"\n{C['bold']}Datos:{C['reset']}")
    dataset_candidates = [
        ROOT / "MondialXboost.Web" / "Data" / "historical_results.csv",
        ROOT / "data" / "raw" / "historical_results.csv",
        ROOT / "data" / "historical_results.csv",
    ]
    dataset_path = next((p for p in dataset_candidates if p.exists()), None)
    if dataset_path:
        print(f"  {C['green']}[OK]{C['reset']} Dataset encontrado: {dataset_path.relative_to(ROOT)}")
    else:
        print(f"  {C['red']}[ERROR]{C['reset']} No se encontró historical_results.csv en ninguna ubicación canónica.")
        issues += 1

    # Models
    print(f"\n{C['bold']}Modelos entrenados:{C['reset']}")
    models_dir = ROOT / "data" / "models"
    for name in ["xgboost_football", "random_forest_football", "cold_start"]:
        if name == "cold_start":
            exists = (models_dir / f"{name}.pkl").exists()
        else:
            exists = (models_dir / f"{name}_meta.json").exists()
        status = C['green'] + "[OK]" if exists else C['yellow'] + "[FALTA]"
        print(f"  {status}{C['reset']} {name}")

    # Write permission
    print(f"\n{C['bold']}Permisos:{C['reset']}")
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        test_file = models_dir / ".doctor_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print(f"  {C['green']}[OK]{C['reset']} data/models es escribible")
    except OSError as exc:
        print(f"  {C['red']}[ERROR]{C['reset']} No se puede escribir en data/models: {exc}")
        issues += 1

    # .NET
    print(f"\n{C['bold']}.NET (opcional para la app web):{C['reset']}")
    dotnet = shutil.which("dotnet")
    if dotnet:
        print(f"  {C['green']}[OK]{C['reset']} dotnet encontrado: {dotnet}")
    else:
        print(f"  {C['yellow']}[WARN]{C['reset']} dotnet no está en PATH (solo necesario para la app Blazor).")
        warnings += 1

    print(f"\n{C['bold']}Resumen:{C['reset']} {C['red']}{issues} error(es){C['reset']}, {C['yellow']}{warnings} advertencia(s){C['reset']}")
    return 1 if issues else 0


def cmd_guia(_args: argparse.Namespace) -> int:
    """Show an extended usage guide with examples."""
    lines = [
        "",
        _color("╔" + "═" * 78 + "╗", "cyan"),
        _color("║" + " Mondial-Xboost — Guía de uso ".center(78) + "║", "cyan"),
        _color("╚" + "═" * 78 + "╝", "cyan"),
        "",
        _color("¿Cómo empezar?", "bold"),
        "  1. Instalá dependencias:            mondial instalar",
        "  2. Entrená el modelo:               mondial entrenar",
        "  3. Predecí un partido:              mondial predecir --home Brazil --away Morocco",
        "",
        _color("Entrenamiento", "bold"),
        "  mondial entrenar                              Entrena el modelo canónico (XGBoost)",
        "  mondial entrenar --name mundial2026           Guarda el modelo con otro nombre",
        "  mondial entrenar --min-date 2015-01-01        Usa solo partidos desde 2015",
        "  mondial entrenar --elo-decay 4                Decaimiento temporal Elo (half-life años)",
        "  mondial entrenar --elo-recent 8.0             Ventana de Elo reciente (años)",
        "  mondial entrenar-gpu                          Entrena usando CUDA",
        "",
        _color("Predicción", "bold"),
        "  mondial predecir --home Argentina --away Brazil",
        "  mondial predecir --home England --away Croatia --blend",
        "  mondial predecir --home Qatar --away Switzerland --cold-start-only",
        "  mondial predecir --home Brazil --away Morocco --model xgboost_football",
        "",
        _color("Optimización y calidad", "bold"),
        "  mondial loop --trials 50                      Tuning de hiperparámetros con Optuna",
        "  mondial loop --trials 100 --auto              10 batches automáticos",
        "  mondial auto-loop                             Ciclo completo: tune + análisis + reentreno + docs",
        "  mondial auto-loop --trials 100 --name exp-04  Experimento personalizado",
        "  mondial auto-loop --trials 50 --backtest      Incluye World Cup backtest",
        "  mondial gates                                 Corre todos los gates de calidad",
        "  mondial backtest                              Backtest sobre World Cup",
        "  mondial auditar                               Revisa leakage temporal",
        "",
        _color("Datos y modelos", "bold"),
        "  mondial manifest                              Muestra el manifest del modelo actual",
        "  mondial elo                                   Compara Elo vs World Football Elo",
        "  mondial data-council                          Revisión del data council",
        "  mondial limpiar                               Borra caché y artefactos de test",
        "",
        _color("Servidor y utilidades", "bold"),
        "  mondial servidor                              Levanta el bridge FastAPI",
        "  mondial servidor-entrenamiento                Levanta dashboard web de entrenamiento (puerto 8765)",
        "  mondial health                                Consulta /health del servidor",
        "  mondial dashboard                             Abre el dashboard ASCII de entrenamiento",
        "  mondial test                                  Ejecuta tests de Python",
        "  mondial lint                                  Ejecuta ruff",
        "  mondial info                                  Muestra información del entorno",
        "  mondial doctor                                Chequeo de portabilidad",
        "",
        _color("Ayuda por comando", "bold"),
        "  mondial entrenar --help",
        "  mondial predecir --help",
        "  mondial loop --help",
        "",
        _color("Consejo:", "yellow") + " si no sabés qué comando usar, ejecutá " + _color("mondial", "cyan"),
        "para ver el menú interactivo.",
        "",
    ]
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


MENU_ITEMS: list[tuple[str, str, list[str]]] = [
    ("Instalar dependencias", "instalar", []),
    ("Entrenar modelo canónico", "entrenar", []),
    ("Entrenar con GPU", "entrenar-gpu", []),
    ("Predecir un partido", "predecir", ["--home", "EQUIPO_LOCAL", "--away", "EQUIPO_VISITANTE"]),
    ("Entrenar modelo cold-start", "entrenar-cold-start", []),
    ("Ejecutar tests", "test", []),
    ("Ejecutar lint", "lint", []),
    ("Verificar gates", "gates", []),
    ("Backtest de World Cup", "backtest", []),
    ("Bridge smoke test", "bridge", []),
    ("Comparar Elo", "elo", []),
    ("Auditar leakage temporal", "auditar", []),
    ("Loop engineering (Optuna)", "loop", []),
    ("Auto Loop Engineering", "auto-loop", []),
    ("Data council", "data-council", []),
    ("Dashboard de entrenamiento", "dashboard", []),
    ("Levantar servidor", "servidor", []),
    ("Dashboard de entrenamiento", "servidor-entrenamiento", []),
    ("Health check", "health", []),
    ("Ver manifest", "manifest", []),
    ("Limpiar artefactos", "limpiar", []),
    ("Info del entorno", "info", []),
    ("Doctor de portabilidad", "doctor", []),
    ("Ver guía de uso", "guia", []),
]


def _show_menu() -> None:
    """Print the interactive menu."""
    print("")
    print(_color("╔" + "═" * 78 + "╗", "cyan"))
    print(_color("║" + " Mondial-Xboost — Menú principal ".center(78) + "║", "cyan"))
    print(_color("╠" + "═" * 78 + "╣", "cyan"))
    print(_color("║" + " Escribí el número y presioná Enter, o 'q' para salir. ".center(78) + "║", "dim"))
    print(_color("╚" + "═" * 78 + "╝", "cyan"))
    print("")
    for idx, (label, command, extra) in enumerate(MENU_ITEMS, start=1):
        extra_str = " ".join(extra) if extra else ""
        line = f"  {idx:2d}) {label:40s}  mondial {command} {extra_str}"
        print(_color(line, "reset"))
    print(f"   0) {_color('Salir', 'yellow')}")
    print("")
    print(_color("También podés usar:", "dim") + " mondial --help  |  mondial guia  |  mondial COMANDO --help")
    print("")


def _prompt_for_args(command: str) -> list[str]:
    """Prompt the user for extra arguments for commands that need them."""
    args: list[str] = []
    if command == "predecir":
        home = input(_color("  Equipo local: ", "cyan")).strip()
        away = input(_color("  Equipo visitante: ", "cyan")).strip()
        if not home or not away:
            print(_color("  [ERROR] Debes ingresar ambos equipos.", "red"))
            return []
        args.extend(["--home", home, "--away", away])
        blend = input(_color("  ¿Usar blended predictor? (s/N): ", "cyan")).strip().lower()
        if blend in {"s", "si", "sí", "yes", "y"}:
            args.append("--blend")
    elif command in {"entrenar", "entrenar-gpu", "entrenar-cold-start"}:
        min_date = input(_color("  Fecha mínima [2010-01-01]: ", "cyan")).strip()
        if min_date:
            args.extend(["--min-date", min_date])
    elif command == "loop":
        trials = input(_color("  Trials [50]: ", "cyan")).strip()
        args.extend(["--trials", trials or "50"])
    return args


def run_menu() -> int:
    """Run the interactive menu loop."""
    parser = build_parser()
    while True:
        _show_menu()
        try:
            choice = input(_color("Seleccioná una opción: ", "bold")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if choice in {"q", "quit", "exit", "0", "salir"}:
            print(_color("¡Hasta luego!", "green"))
            return 0

        if not choice.isdigit():
            print(_color("  [ERROR] Ingresá un número válido.", "red"))
            continue

        idx = int(choice)
        if idx < 1 or idx > len(MENU_ITEMS):
            print(_color("  [ERROR] Opción fuera de rango.", "red"))
            continue

        label, command, extra_template = MENU_ITEMS[idx - 1]
        print("")
        print(_color(f"► {label}", "green"))

        extra_args = _prompt_for_args(command) if any(t in {"EQUIPO_LOCAL", "EQUIPO_VISITANTE"} for t in extra_template) else []
        if extra_template and not extra_args and command == "predecir":
            continue

        argv = [command] + extra_args
        args = parser.parse_args(argv)
        try:
            code = args.func(args)
            if code != 0:
                print(_color(f"\n[FINALIZÓ con código {code}]", "yellow"))
        except Exception as exc:
            print(_color(f"\n[ERROR] {exc}", "red"))

        print("")
        input(_color("Presioná Enter para volver al menú...", "dim"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="mondial",
        description="Mondial-Xboost convenience CLI. Ejecutá 'mondial' sin argumentos para el menú interactivo.",
        epilog="Ejemplos: mondial guia | mondial entrenar --help | mondial predecir --home Brazil --away Morocco",
    )
    subparsers = parser.add_subparsers(dest="command")

    # instalar
    install_parser = subparsers.add_parser(
        "instalar",
        aliases=["install"],
        help="Instala dependencias y el paquete",
        description="Instala las dependencias de Python y el paquete en modo editable.",
        epilog="Ejemplo: mondial instalar",
    )
    install_parser.set_defaults(func=cmd_install)

    # entrenar
    train_parser = subparsers.add_parser(
        "entrenar",
        aliases=["train"],
        help="Entrena el modelo canónico",
        description="Entrena el modelo canónico XGBoost con el dataset histórico.",
        epilog=(
            "Ejemplos:\n"
            "  mondial entrenar\n"
            "  mondial entrenar --name mundial2026\n"
            "  mondial entrenar --min-date 2015-01-01 --elo-decay 4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument("--name", default="xgboost_football", help="Nombre del modelo guardado en data/models/")
    train_parser.add_argument("--min-date", default="2010-01-01", help="Fecha mínima de partidos a incluir (YYYY-MM-DD)")
    train_parser.add_argument("--elo-decay", type=float, default=None, help="Half-life del decaimiento Elo en años (None = sin decaimiento)")
    train_parser.add_argument("--elo-recent", type=float, default=8.0, help="Ventana del Elo reciente en años")
    train_parser.add_argument("--engine", default="xgboost", choices=["xgboost", "random_forest"], help="Motor ML")
    train_parser.set_defaults(func=cmd_train)

    # entrenar-gpu
    train_gpu_parser = subparsers.add_parser(
        "entrenar-gpu",
        aliases=["train-gpu"],
        help="Entrena con GPU (CUDA)",
        description="Entrena el modelo canónico usando GPU CUDA. Requiere xgboost compilado con CUDA.",
        epilog="Ejemplo: mondial entrenar-gpu --name mundial2026",
    )
    train_gpu_parser.add_argument("--name", default="xgboost_football", help="Nombre del modelo")
    train_gpu_parser.add_argument("--min-date", default="2010-01-01", help="Fecha mínima del dataset")
    train_gpu_parser.add_argument("--elo-decay", type=float, default=None, help="Half-life del decaimiento Elo en años")
    train_gpu_parser.add_argument("--elo-recent", type=float, default=8.0, help="Ventana del Elo reciente en años")
    train_gpu_parser.add_argument("--engine", default="xgboost", choices=["xgboost", "random_forest"], help="Motor ML")
    train_gpu_parser.set_defaults(func=cmd_train_gpu)

    # predecir
    predict_parser = subparsers.add_parser(
        "predecir",
        aliases=["predict"],
        help="Predice un partido",
        description="Predice el resultado de un partido entre dos equipos.",
        epilog=(
            "Ejemplos:\n"
            "  mondial predecir --home Brazil --away Morocco\n"
            "  mondial predecir --home Argentina --away Brazil --blend\n"
            "  mondial predecir --home Qatar --away Switzerland --cold-start-only\n"
            "  mondial predecir --home Brazil --away Morocco --engine random_forest --model random_forest_football"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument("--home", required=True, help="Nombre del equipo local")
    predict_parser.add_argument("--away", required=True, help="Nombre del equipo visitante")
    predict_parser.add_argument("--date", default=None, help="Fecha del partido (YYYY-MM-DD, default: hoy)")
    predict_parser.add_argument("--model", default="xgboost_football", help="Nombre del modelo a cargar")
    predict_parser.add_argument("--engine", default="xgboost", choices=["xgboost", "random_forest"], help="Motor ML")
    predict_parser.add_argument("--blend", action="store_true", help="Usar blended predictor (combinación de modelos)")
    predict_parser.add_argument("--cold-start-only", action="store_true", help="Usar solo el modelo de cold-start")
    predict_parser.set_defaults(func=cmd_predict)

    # entrenar-cold-start
    cs_parser = subparsers.add_parser(
        "entrenar-cold-start",
        help="Entrena el modelo de cold-start",
        description="Entrena el modelo fallback para equipos con pocos partidos recientes.",
        epilog="Ejemplo: mondial entrenar-cold-start --cold-threshold 5",
    )
    cs_parser.add_argument("--name", default="cold_start", help="Nombre del modelo")
    cs_parser.add_argument("--min-date", default="2010-01-01", help="Fecha mínima del dataset")
    cs_parser.add_argument("--cold-threshold", type=int, default=10, help="Umbral de partidos recientes para considerar cold-start")
    cs_parser.add_argument("--elo-decay", type=float, default=None, help="Half-life del decaimiento Elo")
    cs_parser.add_argument("--elo-recent", type=float, default=8.0, help="Ventana del Elo reciente")
    cs_parser.set_defaults(func=cmd_train_cold_start)

    # test
    test_parser = subparsers.add_parser("test", help="Ejecuta tests de Python", description="Ejecuta la suite de tests con pytest.")
    test_parser.set_defaults(func=cmd_test)

    # lint
    lint_parser = subparsers.add_parser("lint", help="Ejecuta ruff", description="Ejecuta ruff sobre predictors, scripts, backtest y tests.")
    lint_parser.set_defaults(func=cmd_lint)

    # gates
    gates_parser = subparsers.add_parser("gates", help="Ejecuta verify_gates", description="Corre todos los gates de calidad del pipeline.")
    gates_parser.set_defaults(func=cmd_gates)

    # backtest
    backtest_parser = subparsers.add_parser("backtest", help="Ejecuta el backtest de World Cup", description="Backtest del modelo sobre datos históricos de World Cup.")
    backtest_parser.set_defaults(func=cmd_backtest)

    # bridge
    bridge_parser = subparsers.add_parser("bridge", aliases=["bridge-smoke"], help="Smoke test del bridge C# <-> Python", description="Prueba de humo del bridge entre .NET y Python.")
    bridge_parser.set_defaults(func=cmd_bridge)

    # elo
    elo_parser = subparsers.add_parser("elo", help="Compara Elo contra World Football Elo", description="Compara las ratings Elo del proyecto contra World Football Elo.")
    elo_parser.set_defaults(func=cmd_elo)

    # auditar
    audit_parser = subparsers.add_parser("auditar", aliases=["audit"], help="Audita leakage temporal", description="Auditoría de data leakage temporal en el dataset.")
    audit_parser.set_defaults(func=cmd_audit)

    # loop
    loop_parser = subparsers.add_parser(
        "loop",
        aliases=["tune"],
        help="Tuning de hiperparámetros con Optuna",
        description="Ejecuta el loop engineering para buscar mejores hiperparámetros de XGBoost.",
        epilog="Ejemplos:\n  mondial loop --trials 50\n  mondial loop --trials 100 --auto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    loop_parser.add_argument("--trials", type=int, default=50, help="Trials por batch (default: 50)")
    loop_parser.add_argument("--auto", action="store_true", help="Corre 10 batches automáticamente")
    loop_parser.set_defaults(func=cmd_loop)

    # auto-loop
    auto_loop_parser = subparsers.add_parser(
        "auto-loop",
        aliases=["autoloop", "al"],
        help="Loop engineering automatizado: tune, análisis, estrategia, reentreno y docs",
        description=(
            "Ejecuta un ciclo completo: tuning con Optuna, análisis de resultados, "
            "generación de una estrategia estabilizada, reentrenamiento de un modelo final "
            "y documentación del experimento."
        ),
        epilog=(
            "Ejemplos:\n"
            "  mondial auto-loop\n"
            "  mondial auto-loop --trials 100 --name exp-04-auto-loop\n"
            "  mondial auto-loop --trials 50 --backtest\n"
            "  mondial auto-loop --trials 20 --tune-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    auto_loop_parser.add_argument("--trials", type=int, default=100, help="Trials de Optuna (default: 100)")
    auto_loop_parser.add_argument("--name", type=str, default=None, help="Nombre del experimento y modelo")
    auto_loop_parser.add_argument("--tune-only", action="store_true", help="Solo tuning; no reentrena ni documenta")
    auto_loop_parser.add_argument("--backtest", action="store_true", help="Correr World Cup backtest gate al final")
    auto_loop_parser.add_argument("--no-walk-forward", action="store_true", help="Deshabilitar walk-forward validation")
    auto_loop_parser.set_defaults(func=cmd_auto_loop)

    # data-council
    council_parser = subparsers.add_parser("data-council", aliases=["council"], help="Ejecuta el data council", description="Revisión del data council sobre calidad de datos.")
    council_parser.set_defaults(func=cmd_data_council)

    # dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Levanta el dashboard de entrenamiento", description="Inicia el dashboard de seguimiento de entrenamiento.")
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # servidor
    serve_parser = subparsers.add_parser("servidor", aliases=["serve"], help="Levanta el bridge FastAPI", description="Levanta el servidor FastAPI que hace de bridge entre .NET y Python.")
    serve_parser.set_defaults(func=cmd_serve)

    # servidor-entrenamiento
    training_server_parser = subparsers.add_parser(
        "servidor-entrenamiento",
        aliases=["training-server", "train-server", "ts"],
        help="Levanta el dashboard de monitoreo de entrenamiento",
        description="Levanta un servidor web en http://localhost:8765 para monitorear el entrenamiento en tiempo real.",
        epilog="Ejemplos:\n  mondial servidor-entrenamiento\n  mondial servidor-entrenamiento --port 9000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    training_server_parser.add_argument("--port", type=int, default=None, help="Puerto (default: 8765)")
    training_server_parser.add_argument("--host", type=str, default=None, help="Host (default: 127.0.0.1)")
    training_server_parser.set_defaults(func=cmd_training_server)

    # health
    health_parser = subparsers.add_parser("health", help="Consulta /health del servidor", description="Consulta el endpoint de health del servidor FastAPI.")
    health_parser.set_defaults(func=cmd_health)

    # manifest
    manifest_parser = subparsers.add_parser("manifest", help="Muestra el model_manifest.json", description="Muestra el manifest del modelo actual en formato JSON legible.")
    manifest_parser.set_defaults(func=cmd_manifest)

    # limpiar
    clean_parser = subparsers.add_parser("limpiar", aliases=["clean"], help="Limpia caché y artefactos de test", description="Elimina __pycache__, cachés de pytest/ruff y artefactos de test.")
    clean_parser.set_defaults(func=cmd_clean)

    # info
    info_parser = subparsers.add_parser("info", help="Muestra información del entorno", description="Muestra información del entorno Python, virtualenv y versiones de librerías.")
    info_parser.set_defaults(func=cmd_info)

    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor",
        aliases=["diagnose", "check"],
        help="Verifica portabilidad y salud del entorno",
        description="Ejecuta chequeos de Python, dependencias, dataset, modelos y permisos. Útil al instalar en una PC nueva.",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # guia
    guia_parser = subparsers.add_parser(
        "guia",
        aliases=["guide", "ayuda"],
        help="Muestra la guía de uso con ejemplos",
        description="Muestra una guía detallada de comandos y parámetros con ejemplos.",
    )
    guia_parser.set_defaults(func=cmd_guia)

    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    argv = argv if argv is not None else sys.argv[1:]

    if not argv:
        return run_menu()

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

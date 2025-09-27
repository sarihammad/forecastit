#!/usr/bin/env python3
"""Simple validation script for ForecastIt system."""

import sys
from pathlib import Path


def test_project_structure():
    """Test that the project structure is correct."""
    print("🔍 Testing project structure...")

    required_files = [
        "pyproject.toml",
        "Makefile",
        "README.md",
        "docker-compose.yml",
        "Dockerfile.api",
        "Dockerfile.app",
        ".github/workflows/ci.yml",
        "forecastit/__init__.py",
        "forecastit/config/settings.py",
        "forecastit/data/make_synthetic.py",
        "forecastit/features/build.py",
        "forecastit/modeling/datasets.py",
        "forecastit/api/main.py",
        "forecastit/app/streamlit_app.py",
        "tests/test_data.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_python_syntax():
    """Test Python syntax of key files."""
    print("\n🔍 Testing Python syntax...")

    python_files = [
        "forecastit/config/settings.py",
        "forecastit/data/make_synthetic.py",
        "forecastit/utils/logging.py",
        "forecastit/features/calendar.py",
        "forecastit/modeling/metrics.py",
        "forecastit/api/schemas.py",
    ]

    syntax_errors = []
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                with open(file_path) as f:
                    compile(f.read(), file_path, "exec")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")

    if syntax_errors:
        print(f"❌ Syntax errors found: {syntax_errors}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def test_makefile_targets():
    """Test that Makefile has required targets."""
    print("\n🔍 Testing Makefile targets...")

    required_targets = [
        "install",
        "lint",
        "format",
        "typecheck",
        "test",
        "cov",
        "train",
        "serve-api",
        "serve-app",
        "docker-build",
        "up",
        "down",
        "generate-synth",
    ]

    makefile_path = Path("Makefile")
    if not makefile_path.exists():
        print("❌ Makefile not found")
        return False

    makefile_content = makefile_path.read_text()
    missing_targets = []

    for target in required_targets:
        if f"{target}:" not in makefile_content:
            missing_targets.append(target)

    if missing_targets:
        print(f"❌ Missing Makefile targets: {missing_targets}")
        return False
    else:
        print("✅ All required Makefile targets present")
        return True

def test_docker_config():
    """Test Docker configuration files."""
    print("\n🔍 Testing Docker configuration...")

    docker_files = [
        "docker-compose.yml",
        "Dockerfile.api",
        "Dockerfile.app",
    ]

    missing_files = []
    for file_path in docker_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing Docker files: {missing_files}")
        return False

    # Check docker-compose has required services
    compose_content = Path("docker-compose.yml").read_text()
    required_services = ["api", "app", "mlflow"]

    missing_services = []
    for service in required_services:
        if f"{service}:" not in compose_content:
            missing_services.append(service)

    if missing_services:
        print(f"❌ Missing Docker services: {missing_services}")
        return False
    else:
        print("✅ Docker configuration is complete")
        return True

def test_github_actions():
    """Test GitHub Actions workflow."""
    print("\n🔍 Testing GitHub Actions workflow...")

    workflow_path = Path(".github/workflows/ci.yml")
    if not workflow_path.exists():
        print("❌ GitHub Actions workflow not found")
        return False

    workflow_content = workflow_path.read_text()
    required_jobs = ["lint-and-format", "test", "build-docker"]

    missing_jobs = []
    for job in required_jobs:
        if f"{job}:" not in workflow_content:
            missing_jobs.append(job)

    if missing_jobs:
        print(f"❌ Missing GitHub Actions jobs: {missing_jobs}")
        return False
    else:
        print("✅ GitHub Actions workflow is complete")
        return True

def test_pyproject_toml():
    """Test pyproject.toml configuration."""
    print("\n🔍 Testing pyproject.toml...")

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False

    pyproject_content = pyproject_path.read_text()

    required_sections = [
        "[tool.poetry]",
        "[tool.poetry.dependencies]",
        "[tool.poetry.group.dev.dependencies]",
        "[tool.ruff]",
        "[tool.black]",
    ]

    missing_sections = []
    for section in required_sections:
        if section not in pyproject_content:
            missing_sections.append(section)

    if missing_sections:
        print(f"❌ Missing pyproject.toml sections: {missing_sections}")
        return False
    else:
        print("✅ pyproject.toml is properly configured")
        return True

def main():
    """Run all validation tests."""
    print("🚀 ForecastIt System Validation")
    print("=" * 50)

    tests = [
        test_project_structure,
        test_python_syntax,
        test_makefile_targets,
        test_docker_config,
        test_github_actions,
        test_pyproject_toml,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validations passed! System structure is correct.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: poetry install")
        print('2. Generate synthetic data: poetry run python -c "from forecastit.data.make_synthetic import generate_synthetic_data; generate_synthetic_data()"')
        print("3. Run tests: poetry run pytest tests/")
        print("4. Start API: poetry run uvicorn forecastit.api.main:app --reload")
        print("5. Start dashboard: poetry run streamlit run forecastit/app/streamlit_app.py")
        return 0
    else:
        print("❌ Some validations failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Contributing to tsbk

Thank you for your interest in contributing to tsbk (Triton Server Build Kit)! This guide will help you get started with developing and contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to mlopsplatformteam@grainger.com.

## Getting Started

Before contributing, please:

1. Check existing [issues](https://github.com/your-org/tritonserver-buildkit/issues) to see if your feature/bug has been discussed
2. Open a new issue to discuss significant changes before starting work
3. Fork the repository and create a feature branch for your changes

## Development Environment Setup

### Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **Docker** (for running Triton servers and integration tests)
- **Poetry 1.8.3+** (for dependency management)
- **Git** (for version control)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-user/tritonserver-buildkit.git
   cd tritonserver-buildkit
   ```

2. **Install Poetry**

   If you don't have Poetry installed:

   ```bash
   pipx install poetry==1.8.3
   ```

   Or via pip:

   ```bash
   pip install poetry==1.8.3
   ```

3. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   poetry install --with dev
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

   This will automatically run code quality checks before each commit.

5. **Verify installation**

   ```bash
   tsbk --help
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

   Branch naming conventions:
   - `feature/` - New features
   - `fix/` - Bug fixes
   - `docs/` - Documentation changes
   - `refactor/` - Code refactoring
   - `test/` - Test additions or modifications

2. **Make your changes**

   - Write clean, readable code
   - Follow the existing code style (enforced by black and isort)
   - Add docstrings to new functions and classes
   - Update relevant documentation

3. **Write tests**

   - Add unit tests for new functionality
   - Add integration tests if your changes affect end-to-end workflows
   - Ensure all tests pass before submitting

## Testing

tsbk has two test suites: unit tests and integration tests.

### Unit Tests

Unit tests are fast and don't require external services.

```bash
# Run unit tests with coverage
make unit-tests
```

This command:
- Runs all tests in `tests/unit/`
- Generates a coverage report
- Creates an HTML coverage report in `htmlcov/`

### Integration Tests

Integration tests require Docker and test the full workflow including Triton server deployment.

```bash
# Run integration tests
make integration-tests

# Stop LocalStack when done
docker compose down
```

Integration tests verify:
- Building model repositories from YAML configurations
- Running Triton servers in Docker
- Model inference via HTTP and gRPC
- S3 model artifact fetching
- End-to-end CLI commands

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/unit/test_types.py

# Run a specific test function
pytest tests/unit/test_types.py::test_function_name

# Run with verbose output
pytest -v tests/unit/

# Run with output printed (useful for debugging)
pytest -s tests/unit/
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures (see `conftest.py` files)
- Mock external services in unit tests
- Use descriptive test names that explain what is being tested

## Code Quality

### Formatting and Linting

This project uses several tools to maintain code quality:

- **black**: Code formatter (line length: 121)
- **isort**: Import statement sorter
- **pre-commit**: Automated checks before commits

### Running Code Quality Checks

```bash
# Run all pre-commit hooks manually
pre-commit run --all-files
```

Pre-commit hooks will automatically run when you commit. If they fail:
1. Review the changes made by the hooks
2. Stage the fixed files: `git add .`
3. Commit again

### Code Style Guidelines

- Follow PEP 8 conventions
- Use type hints for function signatures
- Write descriptive variable and function names
- Keep functions focused and single-purpose
- Add docstrings to public functions and classes
- Maximum line length: 121 characters

Example docstring format:

```python
def build_model_repo(config: dict, output_path: str) -> TritonModelRepo:
    """
    Build a Triton model repository from configuration.

    Args:
        config: Dictionary containing model repository configuration
        output_path: Path where the model repository will be created

    Returns:
        TritonModelRepo instance representing the built repository

    Raises:
        ValueError: If configuration is invalid
    """
    ...
```

## Submitting Changes

### Before Submitting

1. Ensure all tests pass:
   ```bash
   make unit-tests
   make integration-tests
   ```

2. Ensure code quality checks pass:
   ```bash
   pre-commit run --all-files
   ```

3. Update documentation if needed:
   - Update README.md for user-facing changes
   - Update docstrings for API changes
   - Add examples if introducing new features

### Creating a Pull Request

1. **Push your branch**

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**

   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template with:
     - **Description**: What changes does this PR introduce?
     - **Motivation**: Why are these changes needed?
     - **Testing**: How were these changes tested?
     - **Breaking Changes**: Does this break existing functionality?
     - **Resolves issues**: Link to any related issues

3. **PR Title Convention**

   Use conventional commit format:
   - `feat: add support for TensorRT optimization`
   - `fix: resolve S3 connection timeout issue`
   - `docs: update installation instructions`
   - `refactor: simplify model loading logic`
   - `test: add integration tests for ensemble models`

### Pull Request Review Process

1. **Automated Checks**: CI will run pre-commit, unit tests, and integration tests
2. **Code Review**: Maintainers will review your code and provide feedback
3. **Address Feedback**: Make requested changes and push updates
4. **Approval**: Once approved, a maintainer will merge your PR

### Tips for Getting PRs Merged

- Keep PRs focused on a single feature or fix
- Write clear commit messages
- Respond promptly to review feedback
- Ensure CI checks pass
- Add tests for new functionality
- Update documentation for user-facing changes

## Release Process

This project uses [semantic-release](https://github.com/semantic-release/semantic-release) for automated versioning and releases.

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Commit Message Format

Commits to the `main` branch should follow conventional commits to trigger releases:

- `feat:` - Triggers a MINOR version bump
- `fix:` - Triggers a PATCH version bump
- `feat!:` or `BREAKING CHANGE:` - Triggers a MAJOR version bump
- `docs:`, `refactor:`, `test:` - No version bump

Example:
```
feat: add support for TensorRT backend

This adds TensorRT backend support with automatic
FP16 optimization capabilities.
```

### How Releases Work

1. PRs are merged to `main`
2. Semantic-release analyzes commit messages
3. Version is automatically bumped in `pyproject.toml` and `src/tsbk/__init__.py`
4. GitHub release is created with changelog
5. Package is published to PyPI

**Note**: Contributors don't need to manually manage versions. The release process is automated based on commit messages.

Thank you for contributing to tsbk!

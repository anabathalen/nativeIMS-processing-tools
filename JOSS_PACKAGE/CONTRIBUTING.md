# Contributing to IMSpartacus

Thank you for your interest in contributing to IMSpartacus! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/imspartacus.git
   cd imspartacus
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below

3. Run tests to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Formatting

We use `black` for code formatting:

```bash
black imspartacus/ --line-length 100
```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest for testing
- Place tests in the `tests/` directory with filenames matching `test_*.py`

Example test:

```python
import pytest
from imspartacus.calibration import CalibrationProcessor

def test_calibration_basic():
    """Test basic calibration functionality."""
    processor = CalibrationProcessor()
    # ... test code ...
    assert result == expected
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Description

Please include:

1. **What** changed
2. **Why** it changed
3. **How** to test the changes
4. Screenshots (if UI changes)
5. Related issue numbers (if applicable)

## Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear and concise description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the behavior
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS and version
   - Python version
   - IMSpartacus version
   - Relevant library versions

6. **Additional Context**: Any other relevant information

## Suggesting Features

When suggesting features:

1. **Description**: Clear description of the feature
2. **Motivation**: Why this feature would be useful
3. **Proposed Solution**: How you envision it working
4. **Alternatives**: Other approaches you've considered
5. **Additional Context**: Any other relevant information

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**

- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Project maintainers are responsible for clarifying standards of acceptable behavior and will take appropriate and fair corrective action in response to unacceptable behavior.

## Questions?

Feel free to:

- Open an issue for discussion
- Contact the maintainers directly
- Join our [community chat/forum if applicable]

## License

By contributing to IMSpartacus, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to IMSpartacus! 🎉

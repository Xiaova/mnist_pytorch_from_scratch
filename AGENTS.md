# AGENTS.md

## 🧠 Project Overview
This is a PyTorch-based deep learning project (MNIST / Fashion-MNIST).
The project is designed for learning and experimentation (CPU-friendly locally, GPU remotely).

---

## ⚙️ Environment Rules (CRITICAL)

- This project ALWAYS runs inside a conda (Anaconda) virtual environment.
- NEVER assume system Python or venv.
- Default environment name:
  
  dl_mnist

- Before running any code, ALWAYS assume:

  conda activate dl_mnist

---

## 📦 Dependency Management (VERY IMPORTANT)

### Priority Order:
1. conda install  ✅ (PRIMARY)
2. pip install    ⚠️ (ONLY if necessary)

### Rules:
- Prefer `conda install` whenever possible.
- Use `pip` ONLY when:
  - package is not available in conda
  - or user explicitly requests pip
- NEVER mix install orders incorrectly:
  - ✅ correct: conda → pip
  - ❌ wrong: pip → conda (may break environment)

### When suggesting installs:
- Always give conda command FIRST
- Then optionally provide pip fallback

### Example:

```bash
conda install numpy
# fallback
pip install numpy
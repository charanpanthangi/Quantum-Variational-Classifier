# Quantum Variational Classifier (QNN) for Simple Binary Classification

## 1. What this project is
This repository shows how to train a quantum variational classifier (QNN) to learn a tiny rule: **y = 1 if x < 0.5 else 0**. The code is intentionally beginner-friendly and pairs the QNN with a classical logistic regression baseline.

## 2. Why this matters
- A QNN behaves like a small neural network neuron but uses quantum gates instead of matrix multiplies.
- The model learns a decision boundary by adjusting rotation gates on a qubit.
- Training is hybrid: a classical optimizer updates the quantum circuit parameters.

## 3. Why SVG files are used instead of PNG
GitHub's CODEX system cannot preview binary files like PNG or JPG and may show a "Binary files are not supported" warning. SVG files are text-based, lightweight, and render directly in diffs, so this repo **only uses SVG** for plots and diagrams.

## 4. How the QNN works
1. Encode the input number `x` as a rotation on the qubit.
2. Apply trainable RX, RY, and RZ gates to shape the state.
3. Measure the Pauli-Z expectation value.
4. Convert that measurement to a probability and classify with a 0.5 threshold.

## 5. How to run the experiment
```bash
pip install -r requirements.txt
python app/main.py --n-samples 150 --epochs 80
```

## 6. Expected output
- QNN accuracy printed to the console.
- Logistic regression accuracy for comparison.
- SVG plots saved in `examples/`:
  - `decision_boundary_qnn.svg`
  - `decision_boundary_classical.svg`
  - `training_loss_qnn.svg`

## Repository layout
- `app/` — dataset generator, QNN model, trainer, plotting helpers, CLI.
- `notebooks/` — walkthrough notebook using SVG output.
- `examples/` — lightweight SVG visuals saved by the CLI.
- `tests/` — pytest checks for dataset, model outputs, and training loop.
- `Dockerfile` — container setup using Python 3.11 slim.
- `requirements.txt` — Python dependencies.

## License
This project is released under the MIT License.

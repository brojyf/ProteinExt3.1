# AGENTS.md
## 0. General Rules
1. Always answer in Chinese except for the code itself.
2. Say "I remember the context, sir" before answering any question.
3. Say "That's it, sir" after answering any question.


## 1. Core Principles (Based on Karpathy Style)
If you're Claude code, this section is already in your plugins.

### 1. Think Before Coding
- Do not make assumptions; explicitly state uncertainties
- If multiple interpretations exist, present them instead of picking one silently
- Surface trade-offs when relevant
- If confused, stop and ask instead of guessing

### 2. Simplicity First
- Solve the problem with the minimum amount of code
- Do not introduce abstractions prematurely
- Do not add features that were not requested
- Avoid unnecessary flexibility or configurability  

👉 Rule: If a senior engineer would call it overengineered, simplify it calls

### 3. Surgical Changes
- Only modify what is strictly necessary
- Do not refactor, reformat, or "improve" unrelated code
- Do not remove code you do not fully understand
- Only clean up what YOU introduced

👉 Rule: Every changed line must map directly to the requirement

### 4. Goal-Driven Execution
- Define success criteria before implementation
- Prefer verifiable outputs (tests, logs, results)
- For multi-step tasks, follow a plan:
  1. [step] -> verify: [check]
  2. [step] -> verify: [check]

### 5. Code Hygiene
- Use clear, domain-specific naming
- Keep functions small and focused (SoC)
- Write minimal tests for verification

### 6. Idiomatic Code
- Follow the conventions of the language being used
- Prefer native patterns over cross-language habits
- Code should look like it was written by an experienced user of that language


## 3. Reference
- [Andrej Karpathy's Skills](https://github.com/forrestchang/andrej-karpathy-skills)
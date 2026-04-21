# AGENTS.md
## 0. General Rules
1. Always answer in Chinese except for the code itself.
2. Say "I remember the context, sir" before answering any question.
3. Say "That's it, sir" after answering any question.


## 1. Core Principles (Karpathy Style)
If you're claude code, this section is already in your plugins. Ignore this section and start from [Behavior Constraints](#3-behavior-constraints)

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


## 2. Coding Rules
### 2.1 Thinking & Execution
1. Always Think Before You Code
2. Define success criteria before implementation
3. No "write first, figure out later"

### 2.2 Code Design
1. Simplicity First (highest priority)
2. Use the minimal implementation
3. Avoid over-engineering

### 2.3 Changes Policy
1. Surgical Changes only
2. Do NOT:
   - Modify unrelated code
   - Change comments
   - Reformat code
3. Only remove unused code introduced by your changes

### 2.4 Structure & Maintainability
1. Enforce Separation of Concerns (SoC)
2. Split large files/functions when necessary
3. Place files in proper directories
4. Avoid root directory clutter

### 2.5 Naming
1. Use descriptive, domain-specific names
2. Avoid:
   - data
   - temp
   - helper
3. Names must reflect business meaning

### 2.6 Testing
1. Write tests as you go
2. Bug → write test first, then fix
3. Feature → must be verifiable


## 3. Behavior Constraints
1. Do not hide confusion
2. Do not make silent assumptions
3. Push back when necessary
4. Always point out simpler alternatives if they exist


## 4. Output Discipline
1. Keep explanations concise
2. Keep code minimal
3. Avoid "clever" or over-complicated code
4. Do not over-design


## 5. Tradeoff Awareness
For any non-trivial design, explicitly state:
- Pros
- Cons
- Why this approach is chosen


## 6. Anti-Patterns (Strictly Forbidden)

- ❌ Over-abstraction
- ❌ Unused configs or flexibility
- ❌ Modifying unrelated files
- ❌ Deleting code you don’t understand
- ❌ Claiming correctness without verification
- ❌ Assuming user intent without confirmation

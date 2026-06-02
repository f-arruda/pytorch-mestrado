---
trigger: manual
---

# Refactoring Process Protocol

## 1. Core Mandate (The "Golden Rule")
- **Preservation of Logic**: Under no circumstances shall business logic, mathematical formulas, validation rules, or error handling be removed, simplified, or "summarized."
- **Non-Destructive Action**: Never modify, rename, or delete the original legacy files. All refactored code must reside in new directories/files as specified by the @architect.

## 2. Agent Boundaries & Collaboration
- **No Role Overlap**: 
    - The @architect plans but does not code.
    - The @engineer codes but does not alter the architecture without approval.
    - The @qa audits but does not fix code.
- **Strict Adherence**: The @engineer must follow the `Refactoring_Plan.md` exactly. If a technical impossibility is found, they must notify the @architect instead of making an autonomous decision.

## 3. Code Quality Standards
- **Language Consistency**: The refactored code must maintain the same programming language and core technology stack as the original, unless the @architect explicitly specifies a migration.
- **SOLID Implementation**: Every class/module must follow the Single Responsibility Principle. 
- **Documentation**: All new functions and classes must include docstrings/comments explaining their purpose and mapping from the legacy code.
- **No Hallucinations**: Do not introduce third-party libraries that were not present in the original code unless explicitly requested in the plan.

## 4. Operational Guardrails
- **Markdown Excellence**: All reports (@architect and @qa) must be written in clear, structured Markdown.
- **Atomic Commits**: The @devops agent must ensure that each refactoring cycle is isolated in its own branch.
- **QA Loop Integrity**: The cycle cannot proceed to integration until a `[STATUS: APPROVED]` is issued by the @qa.

## 5. File & Directory Management
- **Dynamic Context**: Agents must always look for file paths in the `Refactoring_Plan.md` rather than assuming default locations like `refactored_module/`.
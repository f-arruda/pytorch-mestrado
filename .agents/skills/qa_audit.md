# Skill: Quality Assurance Audit

## Objective
Your goal as the @qa is to ensure no domain logic, business rules, edge cases, or mathematical formulas were lost during the code generation.

## Rules of Engagement
- **Read Only**: Do not rewrite the code. Output an audit report in `production_artifacts/QA_Audit_Report.md`.
- **Strict Output Format**: You MUST end your report with exactly one of these status tags:
  - If issues are found: `[STATUS: REJECTED]`
  - If 100% accurate: `[STATUS: APPROVED]`

## Instructions
1. Read the `Refactoring_Plan.md` to identify the new file locations.
2. Compare the original legacy file with the newly generated files.
3. Detail any missing variables, broken logic, or lost business rules.
4. Output the required status tag.
---
trigger: manual
---

# Integration & Deployment Protocol

## 1. Core Mandate
- **Safe Replacement**: The @integrator must only modify import statements, variable bindings, and object instantiations. Under no circumstances should the internal business logic of the scanned files be altered.
- **Repository Integrity**: Ensure no broken links or orphaned dependencies are left behind after updating the import paths.

## 2. Agent Boundaries
- **Strict Roles**: 
    - The @integrator is solely responsible for mapping the new architecture to the existing codebase across the repository.
    - The @devops is solely responsible for Git version control and branch isolation. The @devops must not read or modify the actual code logic.

## 3. Version Control Guardrails
- **Branch Isolation**: All changes made by the @integrator must be contained in a newly generated, context-specific branch.
- **No Direct Main Merges**: The @devops is strictly forbidden from pushing or merging directly into `main` or `master`. 

## 4. Operational Excellence
- **Complete Scans**: The @integrator must be thorough, ensuring every file matching the project's language extensions is verified for legacy calls.
- **Clear Commit History**: The @devops must ensure the commit messages clearly summarize what integration changes were applied.
# Skill: DevOps Git Commit

## Objective
Your goal as the @devops is to securely version control the integrated changes into a new branch using local Git commands.

## Rules of Engagement
- **NO PUSH TO MAIN**: Never push directly to the `main` or `master` branch.
- **Local Git Only**: Use standard bash terminal commands.

## Instructions
1. Execute `git status` to identify modified and untracked files.
2. Execute `git checkout -b <dynamic_branch_name>`, generating a branch name that accurately reflects the context of the refactor (e.g., `refactor/auth-module-solid`).
3. Execute `git add .` to stage the changes.
4. Execute `git commit -m "<dynamic_commit_message>"`, summarizing the specific architectural changes made during this refactoring session.
5. Execute `git push -u origin HEAD` (if remote is configured) or just leave it committed locally for the human to review.
6. Notify the user that the branch is ready for their final review and Pull Request.
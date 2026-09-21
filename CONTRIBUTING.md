# Contributing

## Setup

Follow the [Quick Start](./README.md#quick-start), then enable linting and commit tooling in your environment:

```bash
pre-commit install
pip install commitizen
```

Run the tests with `pytest src/tests`. Architecture rules (layer boundaries, signals, type conversions) are in [src/MVC.md](./src/MVC.md).

## Branching

- **`main`**: production. Every push runs lint and, if the commits warrant it, publishes a versioned release.
- **`staging`**: integration branch. Changes land here before being promoted to `main`.

Don't push directly to either. Instead:

1. Fork the repository and clone your fork.
2. Add the central repo as a remote: `git remote add upstream https://github.com/lanl/annomate-microsentryai-workflow.git`
3. Branch from `staging`: `git checkout staging && git pull upstream staging && git checkout -b feat/my-change`
4. Commit using conventional commits (below), push to your fork, and open a PR against the central repo's **`staging`** branch, not `main`.
5. Once CI passes, a maintainer merges it. Maintainers periodically promote `staging` to `main`, which triggers a release.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<optional scope>): <description>`. Use `cz commit` for an interactive prompt.

| Type | Use | Release |
|---|---|---|
| `fix` | Bug fix | Patch |
| `feat` | New feature | Minor |
| `feat!` / `BREAKING CHANGE:` footer | Breaking change | Major |
| `chore`, `docs`, `style`, `refactor`, `test` | Everything else | None |

Preview the version impact of your commits with `cz bump --dry-run`.

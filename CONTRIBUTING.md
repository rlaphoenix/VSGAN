# Contributing to VSGAN

Thank you for considering contributing to VSGAN!

## Support questions

Please don't use the issue tracker for this. The issue tracker is a tool to address bugs and feature requests in VSGAN
itself. Use one of the following resources for questions about using VSGAN or issues with your own code:

- The [GitHub discussions page] for long term discussion or larger issues.
- For security or other sensitive discussions email me at [rlaphoenix@pm.me].

  [GitHub discussions page]: <https://github.com/rlaphoenix/VSGAN/discussions>
  [rlaphoenix@pm.me]: <mailto:rlaphoenix@pm.me>

## Reporting issues

Include the following information in your post:

- Describe what you expected to happen.
- If possible, include a [minimal reproducible example] to help us identify the issue. This also helps check that the
  issue is not with your own code.
- Describe what actually happened. Include the full traceback if there was an exception.
- List your Python and VSGAN versions, you can provide a commit SHA for VSGAN. If possible, check if this
  issue is already fixed in the latest code in the repository.

  [minimal reproducible example]: <https://stackoverflow.com/help/minimal-reproducible-example>

## Submitting patches

If there is not an open issue for what you want to submit, prefer opening one for discussion before working on a PR.
You can work on any issue that doesn't have an open PR linked to it or a maintainer assigned to it. These show up in
the sidebar. No need to ask if you can work on an issue that interests you.

Include the following in your patch:

- Make sure you test your changes or new additions.
- Update any relevant docs pages and docstrings. Doc pages and docstrings should be wrapped at most 116 characters.

### First time setup

The entire process of setting up your Development environment to work on VSGAN can be found on the [Building]
documentation. It guides you through the download, preparation, and installation of the source code.
I recommend working on the source code through Git, and from a [Fork] of the repository.

  [Building]: <https://vsgan.phoeniix.dev/en/stable/building.html>
  [Fork]: <https://github.com/rlaphoenix/VSGAN/fork>

### Start coding

- Don't forget to install [pre-commit] so your commits are checked for code syntax, format, and quality.
- Create a branch to identify the issue you would like to work on. Do not work directly off the main/master branch.
  Branch off of the origin/master branch for the latest upstream changes.

```bash
$ git fetch origin
$ git checkout -b your-branch-name origin/main
```

- Using your favorite editor, make your changes, [committing as you go][Please Commit!!].
- Make sure you fully test all changes and additions from as many directions and edge cases as possible.
- Changes must pass a MyPy and Flake8 test with no errors and minimal to no warnings.
- Push your commits to your fork on GitHub and [create a pull request][Creating a PR]. Link to the issue being
  addressed with `fixes #123` in the pull request to automatically link the issue and mark the PR as a fix.

```bash
$ git push --set-upstream fork your-branch-name
```

  [pre-commit]: <https://pre-commit.com>
  [Please Commit!!]: <https://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html#commit-your-changes>
  [Creating a PR]: <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>

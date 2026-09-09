# Getting help

**Something does not work on your machine?** Run this and paste the output into a bug report
(remove anything you consider private, such as paths with your name):

```
llamafit --json doctor
```

It lists every detection step, whether it succeeded, and what LlamaFit would do differently
with more information. Most "it does not see my GPU" reports are answered by the hint next to
the failed probe.

**A model is missing or its data is wrong?** Open a *Model request* issue. The catalog is
curated by hand from primary sources, and corrections with a link to the source are merged
quickly.

**A question rather than a bug?** Use GitHub Discussions on this repository. Questions about
llama.cpp itself (build flags, model conversion) are better asked in the llama.cpp project.

**Security concerns:** see [SECURITY.md](SECURITY.md); do not open a public issue.

## Before reporting a bug

1. Update: `pip install -U llamafit`.
2. Check the [platform support](docs/platform-support.md) page for the probe that failed.
3. Search existing issues; add your `doctor` output to a matching one instead of opening a
   duplicate.

"""The placement planner: where a model's bytes go, and the settings that put them there.

Section 9 of the design. What is *not* here is the budget: sizing a configuration is
section 8's job, and this package takes it as an injected callable --
:class:`~llamafit.placement.modes.BudgetFn` -- the way the rest of the project injects its
command runner and its HTTP client, so the two can be read, tested and changed apart.
"""

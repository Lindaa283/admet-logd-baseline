# Makes this a package so test module names are qualified
# (logd_suite.test_metrics), which keeps them from colliding with same-named
# test files belonging to the parallel src/admet work.
#
# Named logd_suite, not logd: a test package called `logd` sits on sys.path
# ahead of src/logd and shadows it, so `from logd import config` resolves to
# this __init__.py and every import fails.

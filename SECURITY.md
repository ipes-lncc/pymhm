# Security policy

The package is pre-alpha. Security fixes target the latest development revision
and the latest release, when one is available.

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/ipes-lncc/pymhm/security/advisories/new).
If that feature is unavailable, contact `volpatto@lncc.br`. Include affected
versions, a minimal reproducer, impact, and any proposed remediation. Do not
publish sensitive details before maintainers have had an opportunity to respond.

Treat user-provided Python callbacks, UFL forms, and notebooks as executable
code. The package does not sandbox them.

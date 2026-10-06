# Security policy

Security fixes target the latest official release and the latest development
revision.

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/ipes-lncc/pymhm/security/advisories/new).
If that feature is unavailable, contact `volpatto@lncc.br`. Include affected
versions, a minimal reproducer, impact, and any proposed remediation. Do not
publish sensitive details before maintainers have had an opportunity to respond.

Treat user-provided Python callbacks, UFL forms, and notebooks as executable
code. The package does not sandbox them.

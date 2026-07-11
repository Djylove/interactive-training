# Security

## Reporting

Please report vulnerabilities privately to the corresponding authors listed in the
paper before opening a public issue.

## Credentials

Do not commit API keys, Hugging Face tokens, W&B credentials, `.env` files, cluster
environment scripts, or generated prompts containing confidential context.

If a credential has ever appeared in git history, deleting the current file is not
sufficient. Revoke and rotate the credential, remove it from all reachable history,
and invalidate cached artifacts before publishing the repository.

Interactive Training accepts LLM credentials as write-only runtime configuration.
The `/state` endpoint exposes only `api_key_set`, and action logging redacts the
`api_key` field. These safeguards do not protect credentials printed by third-party
libraries or inserted into custom event payloads.

## Agent authority

The supplied LLM agent cannot load checkpoints, pause/resume training, reset modules,
change context, or reconfigure itself. Applications remain responsible for:

- validating custom action payloads;
- setting safe value bounds and rate limits;
- restricting network exposure of the HTTP control endpoint;
- adding approval gates and rollback for high-impact actions; and
- limiting accelerator, API, and storage budgets.

The default HTTP server is intended for trusted local or cluster networks and does not
implement authentication or TLS. Do not expose it directly to the public internet.

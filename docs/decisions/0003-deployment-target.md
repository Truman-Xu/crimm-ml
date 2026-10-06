# 0003 — Deploy on one small VM first; serverless later

*Accepted, 2026-10-06.*

## Context

The first public deployment is a CPU REST API in a container, expected to
handle one or two jobs at a time, each about 50 s of computation (T=50, L≈80,
8 threads). The candidates were a small virtual machine, a serverless
container platform (Cloud Run), and Kubernetes.

## Decision

One 4 vCPU / 16 GB VM running the container under rootless podman, supervised
by systemd through a Quadlet unit, behind a Caddy reverse proxy for TLS.

## Reasons

- **Serverless containers throttle CPU outside a request.** The job API
  answers `202 Accepted` at once and computes afterwards, with no request in
  flight. Cloud Run cuts CPU to near zero in that state unless CPU is "always
  allocated" with a minimum of one instance — which costs as much as a VM and
  removes the advantage.
- **Scale to zero loses work.** An instance shut down while idle takes
  in-flight jobs and the in-memory job table with it.
- **Cold starts are as long as the job.** 25–70 s to start an instance and
  load a 275 MB model, to serve a 50 s job, is worse than a short queue on a
  warm machine.
- **Kubernetes solves problems this service does not have.** It pays off with
  many services, many nodes, rolling deploys and self-healing; for one service
  it adds a control plane and a large operational surface for nothing.

## Consequences

- One machine is one failure domain; jobs in flight are lost on restart. That
  is acceptable for a free, best-effort service and stated in the API docs.
- Results expire after 7 days (a systemd timer); nothing here is a database,
  so nothing needs backups.

## Revisit when

The job table and result storage move out of the process (`JobStore` on Redis
or Postgres, `ObjectStore` on object storage). Then the right shape is a queue
(Cloud Tasks) feeding a serverless *worker* service with concurrency 1, where
each job is a request and CPU throttling no longer applies.

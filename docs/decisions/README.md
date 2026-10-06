# Decision records

Short, dated records of choices that shaped this project, with the evidence at
the time. When a decision is revisited, add a new record that supersedes the
old one rather than editing it, so the reasoning stays traceable.

| # | Decision | Status |
|---|---|---|
| [0001](0001-onnx-not-tensorflow.md) | Ship the ONNX export; no TensorFlow port, no TPU | accepted |
| [0002](0002-process-per-job.md) | The service runs each job in its own worker process | accepted |
| [0003](0003-deployment-target.md) | Deploy on one small VM first; serverless later | accepted |

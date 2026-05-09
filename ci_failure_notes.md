# CI failure notes for PR #23862

PR: https://github.com/sgl-project/sglang/pull/23862

## Status

Skipped CI retry for now. The failing Nvidia PR Test includes a real test-body failure in `stage-b-test-4-gpu-b200`, not just an AMD/NPU signal.

## Failing job

- Workflow run: https://github.com/sgl-project/sglang/actions/runs/25330912247
- Job: `stage-b-test-4-gpu-b200`
- Runner: `b200-di01-0123`
- Failed step: `Run test`
- Failed test file: `test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py`

## Error

The server process exits during setup for `TestDeepseekV3FP4MTP`.

Key traceback:

```text
File "/actions-runner/_work/sglang/sglang/python/sglang/srt/models/deepseek_nextn.py", line 185, in forward
    forward_batch.spec_info.hidden_states
AttributeError: 'NoneType' object has no attribute 'hidden_states'

Exception: Server process exited with code -9. Check server logs for errors.
FAILED: /actions-runner/_work/sglang/sglang/test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py returned exit code 1
```

There is also a JIT cache race-looking error earlier:

```text
FileNotFoundError: [Errno 2] No such file or directory:
/root/.cache/tvm-ffi/sgl_kernel_jit_activation_bf16_t_true_92ff72c75a8dd73f
```

The later `deepseek_nextn.py` failure is the stronger code signal because this PR changes EAGLE draft-model KV-cache accounting.

## Possible fix direction

Audit the EAGLE path changed by this PR to ensure `forward_batch.spec_info` is initialized before the DeepSeek NextN draft model forward path reads `spec_info.hidden_states`.

Likely areas to inspect:

- `python/sglang/srt/speculative/eagle_worker.py`
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`
- `python/sglang/srt/models/deepseek_nextn.py`

The likely fix is to preserve or initialize the speculative `hidden_states` metadata when accounting for the draft model KV cache, instead of allowing the NextN forward path to see `forward_batch.spec_info is None`.

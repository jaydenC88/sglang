cd python
# SGLANG_SPEC_NAN_DETECTION=1 SGLANG_SPEC_OOB_DETECTION=1 \
uv run sglang serve --model-path meta-llama/Llama-2-7b-chat-hf \
--speculative-algorithm=EAGLE \
--speculative-draft-model-path=lmsys/sglang-EAGLE-llama2-chat-7B \
--speculative-num-steps=5 \
--speculative-eagle-topk=8 \
--speculative-num-draft-tokens=64 \
--chunked-prefill-size 128 \
--max-running-requests 8 \
--max-running-requests 8 --device cuda --host 127.0.0.1 --port 11000 \
--disable-piecewise-cuda-graph 2>&1 | tee ../log.txt

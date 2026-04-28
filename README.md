# nousnet-diagrams

Public diagrams and visual notes for the [nousnet](https://github.com/NousResearch/nousnet) library.

## Multi-LoRA per-row routing

How K LoRA adapters share one forward pass over a packed batch — each row is routed to its own `(lora_A, lora_B)` pair via gather/scatter, while the base linear runs once for everyone. Autograd separates the gradients automatically because each adapter's weights are only invoked on the rows belonging to that adapter.

![multi-lora routing](images/multi_lora_routing.png)

**Key idea**: 1 forward pass, 1 base matmul shared across all rows, K small adapter pairs each touching only its own rows. `∂L/∂(lora_A[i])` is non-zero only on rows where `adapter_ids == i`. The base sees the sum across all rows.

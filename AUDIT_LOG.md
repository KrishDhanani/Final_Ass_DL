# Incident Audit Log

## Summary
This document itemizes every bug, corruption, and anti-pattern discovered in the recovered codebase, along with the fixes applied.

| Bug ID | File | Problem | Root Cause | Fix Applied | Commit Hash |
|--------|------|---------|-----------|-------------|-------------|
| BUG-001 | models.py (AlexNet) | Shape mismatch: expected 2048, got 3072 | Hardcoded classifier input size didn't match flattened feature dimensions (4×4×192=3072) : Conv2d/MaxPool2d:output_size = (input_size - kernel_size + 2*padding) / stride + 1 | Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)` | N/A |
| BUG-002 | models.py (AlexNet) | Last layer hardcoded to 11 classes | Output layer: `nn.Linear(1024, 11)` instead of `num_classes` as outputsize | Changed to `nn.Linear(1024, kwargs.get("num_classes", 2))` | N/A |
| BUG-003 | fit.py | 0D or 1D target tensor expected, multi-target not supported: outputs shape: torch.Size([32, 2]), labels shape: torch.Size([32, 1])  | The “NUM_CLASSES”=2, so at the end output of shape is [32,2]. when data goes into the BinaryCrossEntropy then still need to convert from [32,2] -> [32]. Require Squeezing.`.squeeze()` removes all size-1 dimensions, breaks on [1,1] → scalar | Changed :: labels.squeeze()` to `labels.squeeze(1)` in both train & eval for Error `Expected batch_size (1) to match target batch_size (0)" — the last validation batch broke!| N/A |
| BUG-004 | train.py, config.json | Wrong NUM_CLASSES configuration | Config hardcoded to 2, but cells data has 8 classes | Updated config with correct class counts per dataset | N/A |
| BUG-005 | fit.py | Gradients explode across batches | Missing `optimizer.zero_grad()` accumulates gradients Also when tried to train model the accuracy too low ~18%(Alexnet)| Added `self.optimizer.zero_grad()` before `loss.backward()` | N/A |
| BUG-006 | train.py | Model learns at random-guessing level (~19% accuracy) | `drop_rate=0.99` disables 99% of neurons | Changed `drop_rate=0.99` to `drop_rate=0.5` | N/A |
| BUG-007 | models.py (AlexNet, VGG16, ResNet18) | Grayscale data (chest, orgs) crashes on hardcoded 3 channels | First Conv2d hardcoded `in_channels=3`, but grayscale is 1 | Made `in_channels` dynamic: `in_channels = kwargs.get("in_channels", 3)` | N/A |
| BUG-008 | models.py (VGGBlock) | Channel mismatch in sequential convolutions | `current_in_channels` never updated after first Conv2d | Added `current_in_channels = out_channels` after each conv | N/A |
| BUG-009 | models.py (VGG16, AlexNet) | Classifier input size hardcoded to 2048 | Different datasets produce different flattened sizes | Implemented dynamic size calculation using dummy input | N/A |
| BUG-010 | models.py (ResNet18) | Forward returns None | Missing `return` statement in `forward()` method | Added `return self.classifier(out)` | N/A |
| BUG-011 | models.py (ResNet18) | Very slow learning with Identity activation | `activation_str = "Identity"` makes network essentially linear | Changed `activation_str = "Identity"` to `activation_str = "ReLU"` | N/A |

---


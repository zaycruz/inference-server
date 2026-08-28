import sys
sys.path.insert(0, "/root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
from vllm.models.glm5next import Glm5NextForConditionalGeneration
print("class import OK:", Glm5NextForConditionalGeneration.__name__)

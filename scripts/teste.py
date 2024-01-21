from omnixai.explainers.vision import VisionExplainer
from omnixai.visualization.dashboard import Dashboard

explainer = VisionExplainer(
    explainers=["gradcam", "lime", "ig", "ce", "feature_visualization"],
    mode="classification",
    model=model,                   # An image classification model, e.g., ResNet50
    preprocess=preprocess,         # The preprocessing function
    postprocess=postprocess,       # The postprocessing function
    params={
        # Set the target layer for GradCAM
        "gradcam": {"target_layer": model.layer4[-1]},
        # Set the objective for feature visualization
        "feature_visualization": 
          {"objectives": [{"layer": model.layer4[-3], "type": "channel", "index": list(range(6))}]}
    },
)
# Generate explanations of GradCAM, LIME, IG and CE
local_explanations = explainer.explain(test_img)
# Generate explanations of feature visualization
global_explanations = explainer.explain_global()
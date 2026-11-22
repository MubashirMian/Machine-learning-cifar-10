# """SimCLR CIFAR-10 Visual Representation Inference Web Application.
# Inspired by Stack Overflow design aesthetics.
# Deploys on Hugging Face Spaces (Gradio SDK).
# """
# import os
# import sys
# from pathlib import Path
# import time
# from typing import Dict, Tuple

# # Enable importing local model_arch whether executed from app/ or project root
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import gradio as gr
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from torchvision import transforms

# from model_arch import LinearClassifier, ResNet18CIFAR

# # CIFAR-10 normalization constants
# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CLASSES = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck"
# ]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model weights
# MODEL_PATH = Path("simclr_cifar10_eval.pt")
# if not MODEL_PATH.is_file():
#     MODEL_PATH = Path("app/simclr_cifar10_eval.pt")

# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# encoder = ResNet18CIFAR().to(DEVICE)
# encoder.load_state_dict(checkpoint["encoder_state_dict"])
# encoder.eval()

# classifier = LinearClassifier(in_features=512, num_classes=10).to(DEVICE)
# classifier.load_state_dict(checkpoint["classifier_state_dict"])
# classifier.eval()

# eval_transform = transforms.Compose([
#     transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
# ])


# def predict(image: Image.Image) -> Tuple[Dict[str, float], str]:
#     if image is None:
#         return {}, "<div class='so-box'>Please upload an image or choose one of the sample examples below.</div>"

#     t0 = time.time()
#     img_rgb = image.convert("RGB")
#     tensor = eval_transform(img_rgb).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         features = encoder(tensor)
#         logits = classifier(features)
#         probs = F.softmax(logits, dim=1).squeeze(0)

#     latency_ms = (time.time() - t0) * 1000.0

#     confidences = {CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))}
#     top_pred = CLASSES[probs.argmax().item()]
#     top_conf = probs.max().item() * 100.0

#     l2_norm = float((features ** 2).sum().sqrt().item())
#     feat_mean = float(features.mean().item())
#     feat_std = float(features.std().item())
#     is_finite = bool(torch.isfinite(features).all())

#     stats_html = f"""
#     <div class="so-card" style="margin-top: 15px;">
#         <div class="so-header-sub">
#             <span class="so-badge-orange">PREDICTION RESULT</span>
#             <span style="float: right; color: #6A737C; font-size: 13px;">Latency: <strong>{latency_ms:.1f} ms</strong></span>
#         </div>
#         <div style="padding: 15px;">
#             <div style="display: flex; align-items: center; margin-bottom: 12px;">
#                 <div class="so-vote-box" style="margin-right: 15px;">
#                     <div style="font-size: 22px; font-weight: bold; color: #0C0D0E;">{top_conf:.1f}%</div>
#                     <div style="font-size: 11px; color: #6A737C; text-transform: uppercase;">Confidence</div>
#                 </div>
#                 <div>
#                     <div style="font-size: 20px; font-weight: 700; color: #F48024; text-transform: uppercase;">
#                         {top_pred}
#                     </div>
#                     <div style="font-size: 12px; color: #525960;">
#                         Predicted class from 10 CIFAR-10 benchmark visual categories
#                     </div>
#                 </div>
#             </div>
            
#             <div style="border-top: 1px solid #E3E6E8; padding-top: 12px;">
#                 <div style="font-size: 13px; font-weight: 600; color: #232629; margin-bottom: 6px;">Representation Metrics (h &isin; &reals;<sup>512</sup>)</div>
#                 <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; font-family: monospace;">
#                     <div>&bull; Embedding Dimension: <strong>512-d</strong></div>
#                     <div>&bull; L2-Norm: <strong>{l2_norm:.4f}</strong></div>
#                     <div>&bull; Embedding Mean: <strong>{feat_mean:.4f}</strong></div>
#                     <div>&bull; Embedding Std: <strong>{feat_std:.4f}</strong></div>
#                     <div>&bull; Finiteness: <strong style="color: {'#2F6F44' if is_finite else '#D0393E'};">{'VALID' if is_finite else 'NaN/Inf'}</strong></div>
#                     <div>&bull; Device: <strong>{DEVICE}</strong></div>
#                 </div>
#             </div>
#         </div>
#     </div>
#     """

#     return confidences, stats_html


# SAMPLE_DIR = Path("samples")
# if not SAMPLE_DIR.is_dir():
#     SAMPLE_DIR = Path("app/samples")

# examples = []
# if SAMPLE_DIR.is_dir():
#     examples = [[str(p)] for p in sorted(SAMPLE_DIR.glob("*.png"))]

# custom_css = """
# /* Stack Overflow Aesthetic Theme */
# body, .gradio-container {
#     background-color: #F8F9F9 !important;
#     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
#     color: #232629 !important;
# }

# .so-topbar {
#     background-color: #232629;
#     border-top: 3px solid #F48024;
#     padding: 12px 20px;
#     display: flex;
#     align-items: center;
#     color: #FFFFFF;
#     margin-bottom: 20px;
#     border-radius: 4px;
# }

# .so-topbar-title {
#     font-size: 18px;
#     font-weight: 700;
#     color: #FFFFFF;
#     display: flex;
#     align-items: center;
#     gap: 8px;
# }

# .so-tag {
#     display: inline-block;
#     padding: 3px 6px;
#     font-size: 12px;
#     color: #39739D;
#     background-color: #E1ECF4;
#     border-radius: 3px;
#     margin-right: 4px;
#     margin-bottom: 4px;
#     text-decoration: none;
#     font-weight: 500;
# }

# .so-card {
#     background-color: #FFFFFF;
#     border: 1px solid #E3E6E8;
#     border-radius: 6px;
#     box-shadow: 0 1px 3px rgba(0,0,0,0.06);
#     margin-bottom: 16px;
# }

# .so-header-sub {
#     background-color: #F8F9F9;
#     border-bottom: 1px solid #E3E6E8;
#     padding: 10px 15px;
#     font-size: 13px;
#     font-weight: 600;
#     color: #525960;
# }

# .so-badge-orange {
#     background-color: #F48024;
#     color: #FFFFFF;
#     padding: 2px 7px;
#     border-radius: 3px;
#     font-size: 11px;
#     font-weight: 700;
# }

# .so-badge-green {
#     background-color: #E1F5E6;
#     color: #2F6F44;
#     border: 1px solid #C3E6CB;
#     padding: 3px 8px;
#     border-radius: 3px;
#     font-size: 12px;
#     font-weight: 600;
# }

# .so-vote-box {
#     text-align: center;
#     padding: 8px 12px;
#     border: 1px solid #E3E6E8;
#     border-radius: 5px;
#     background-color: #FDF7E7;
# }

# .so-sidebar-box {
#     background-color: #FDF7E7;
#     border: 1px solid #F1E5BC;
#     border-radius: 5px;
#     padding: 12px 14px;
#     margin-bottom: 15px;
#     font-size: 13px;
# }

# .so-sidebar-title {
#     font-weight: 700;
#     color: #0C0D0E;
#     margin-bottom: 8px;
#     font-size: 13px;
#     border-bottom: 1px solid #F1E5BC;
#     padding-bottom: 4px;
# }

# .so-btn-orange {
#     background-color: #F48024 !important;
#     color: white !important;
#     border: none !important;
#     font-weight: 600 !important;
# }

# footer {
#     display: none !important;
# }
# """

# with gr.Blocks(css=custom_css, title="StackSimCLR — CIFAR-10 Visual Representation Classifier") as demo:
#     # Stack Overflow Top Bar
#     gr.HTML("""
#     <div class="so-topbar">
#         <div class="so-topbar-title">
#             <span style="color: #F48024; font-size: 22px;">&#9776;</span>
#             <span style="color: #FFFFFF;">stack</span><span style="color: #F48024; font-weight: 800;">simclr</span>
#         </div>
#         <div style="margin-left: 20px; font-size: 13px; color: #BABFC4;">
#             Self-Supervised Visual Representation & Classification on CIFAR-10
#         </div>
#         <div style="margin-left: auto; display: flex; gap: 8px;">
#             <span class="so-badge-green">&check; S6 Reproduction: 75.23% Top-1</span>
#             <span class="so-badge-green">&check; S7 Scaling Evaluated</span>
#         </div>
#     </div>
#     """)

#     # Question Header Block
#     gr.HTML("""
#     <div style="margin-bottom: 15px;">
#         <h1 style="font-size: 24px; font-weight: 600; color: #232629; margin-bottom: 6px;">
#             How does SimCLR learn visual representations on CIFAR-10 from scratch without labels?
#         </h1>
#         <div style="font-size: 12px; color: #6A737C; border-bottom: 1px solid #E3E6E8; padding-bottom: 12px;">
#             <span>Asked: <strong>Course Project</strong></span>
#             <span style="margin-left: 15px;">Paper: <strong>Chen et al. (ICML 2020)</strong></span>
#             <span style="margin-left: 15px;">Reproduction Status: <strong>VERIFIED (Stage S6 & S7 COMPLETE)</strong></span>
#             <span style="margin-left: 15px;">Encoder: <strong>ResNet-18 (CIFAR stem)</strong></span>
#         </div>
#         <div style="margin-top: 8px;">
#             <span class="so-tag">pytorch</span>
#             <span class="so-tag">simclr-v1</span>
#             <span class="so-tag">cifar-10</span>
#             <span class="so-tag">contrastive-learning</span>
#             <span class="so-tag">resnet-18</span>
#             <span class="so-tag">self-supervised</span>
#         </div>
#     </div>
#     """)

#     with gr.Row():
#         # Main Column: Upload & Prediction
#         with gr.Column(scale=3):
#             gr.HTML("""
#             <div class="so-card">
#                 <div class="so-header-sub">
#                     <strong>1. INPUT IMAGE (TEST BENCHMARK)</strong> &mdash; Upload an image or select a sample
#                 </div>
#             </div>
#             """)
#             input_image = gr.Image(type="pil", label="Input Image (Auto-resized & normalized)", height=220)

#             predict_btn = gr.Button("Classify with SimCLR Representation", variant="primary", elem_classes=["so-btn-orange"])

#             output_label = gr.Label(num_top_classes=5, label="Top-5 Classification Confidences")
#             output_stats = gr.HTML()

#             if examples:
#                 gr.Examples(examples=examples, inputs=input_image, label="CIFAR-10 Test Sample Examples (Click to Test)")

#         # Right Column: The Overflow Blog / Project Context
#         with gr.Column(scale=2):
#             gr.HTML("""
#             <div class="so-sidebar-box">
#                 <div class="so-sidebar-title">The Overflow Blog: SimCLR v1</div>
#                 <div style="line-height: 1.5; color: #3B4045;">
#                     <p style="margin-bottom: 8px;">
#                         <strong>What is this app?</strong><br>
#                         This system demonstrates <strong>Contrastive Visual Representation Learning</strong>. 
#                         The base ResNet-18 model was pretrained on unlabeled CIFAR-10 images using the <strong>NT-Xent contrastive loss</strong>.
#                     </p>
#                     <p style="margin-bottom: 8px;">
#                         <strong>Key Architecture:</strong><br>
#                         &bull; <strong>Stem:</strong> 3&times;3 conv (stride 1), no max-pool (preserves 32&times;32 resolution).<br>
#                         &bull; <strong>MLP Head:</strong> 512 &rarr; 512 &rarr; 128 (discarded after pretrain).<br>
#                         &bull; <strong>Representation:</strong> Vector <em>h</em> &isin; &reals;<sup>512</sup> fed to downstream frozen linear probe.
#                     </p>
#                 </div>
#             </div>

#             <div class="so-sidebar-box" style="background-color: #F2F8FD; border-color: #D3E5F5;">
#                 <div class="so-sidebar-title" style="border-color: #D3E5F5; color: #1B4F72;">Stage S7: Data Scaling Curve</div>
#                 <div style="font-size: 12px; line-height: 1.4;">
#                     Our data fraction experiment verified monotonic scaling with diminishing returns:
#                     <table style="width: 100%; margin-top: 6px; font-size: 11px; border-collapse: collapse;">
#                         <tr style="border-bottom: 1px solid #D3E5F5;">
#                             <th style="text-align: left; padding: 2px;">Fraction</th>
#                             <th style="text-align: right; padding: 2px;">Samples</th>
#                             <th style="text-align: right; padding: 2px;">Top-1 Acc</th>
#                         </tr>
#                         <tr><td>10%</td><td style="text-align: right;">4,500</td><td style="text-align: right;"><strong>54.17%</strong></td></tr>
#                         <tr><td>25%</td><td style="text-align: right;">11,250</td><td style="text-align: right;"><strong>65.37%</strong></td></tr>
#                         <tr><td>50%</td><td style="text-align: right;">22,500</td><td style="text-align: right;"><strong>71.04%</strong></td></tr>
#                         <tr style="background-color: #E1F5E6;"><td><strong>100%</strong></td><td style="text-align: right;">45,000</td><td style="text-align: right;"><strong>75.23%</strong></td></tr>
#                     </table>
#                 </div>
#             </div>

#             <div class="so-card">
#                 <div class="so-header-sub">PROJECT ATTRIBUTION</div>
#                 <div style="padding: 12px; font-size: 12px; line-height: 1.5; color: #525960;">
#                     <div><strong>Course:</strong> Machine Learning Paper Reproduction</div>
#                     <div><strong>Institution:</strong> Institute of Business Administration (IBA), Karachi</div>
#                     <div style="margin-top: 6px;">
#                         <strong>Team Members:</strong><br>
#                         &bull; Saalik Hussain Zaidi (36206)<br>
#                         &bull; Mubashir Abid (36224)<br>
#                         &bull; Muhammad Zohair Khan (33939)
#                     </div>
#                 </div>
#             </div>
#             """)

#     try:
#         import spaces
#         predict_fn = spaces.GPU(predict)
#     except Exception:
#         predict_fn = predict

#     predict_btn.click(fn=predict_fn, inputs=input_image, outputs=[output_label, output_stats])
#     input_image.change(fn=predict_fn, inputs=input_image, outputs=[output_label, output_stats])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)

"""SimCLR CIFAR-10 Visual Representation Inference Web Application.
Inspired by Stack Overflow design aesthetics.
Deploys on Hugging Face Spaces (Gradio SDK).
"""
import os
import sys
from pathlib import Path
import time
from typing import Dict, Tuple

# Enable importing local model_arch whether executed from app/ or project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from model_arch import LinearClassifier, ResNet18CIFAR

# ZeroGPU support: decorate at definition time; fall back to a no-op elsewhere
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
MODEL_PATH = Path("simclr_cifar10_eval.pt")
if not MODEL_PATH.is_file():
    MODEL_PATH = Path("app/simclr_cifar10_eval.pt")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

encoder = ResNet18CIFAR().to(DEVICE)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
encoder.eval()

classifier = LinearClassifier(in_features=512, num_classes=10).to(DEVICE)
classifier.load_state_dict(checkpoint["classifier_state_dict"])
classifier.eval()

eval_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


@gpu
def predict(image: Image.Image) -> Tuple[Dict[str, float], str]:
    if image is None:
        return {}, "<div class='so-box'>Please upload an image or choose one of the sample examples below.</div>"

    t0 = time.time()
    img_rgb = image.convert("RGB")
    tensor = eval_transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(tensor)
        logits = classifier(features)
        probs = F.softmax(logits, dim=1).squeeze(0)

    latency_ms = (time.time() - t0) * 1000.0

    confidences = {CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))}
    top_pred = CLASSES[probs.argmax().item()]
    top_conf = probs.max().item() * 100.0

    l2_norm = float((features ** 2).sum().sqrt().item())
    feat_mean = float(features.mean().item())
    feat_std = float(features.std().item())
    is_finite = bool(torch.isfinite(features).all())

    stats_html = f"""
    <div class="so-card" style="margin-top: 15px;">
        <div class="so-header-sub">
            <span class="so-badge-orange">PREDICTION RESULT</span>
            <span style="float: right; color: #6A737C; font-size: 13px;">Latency: <strong>{latency_ms:.1f} ms</strong></span>
        </div>
        <div style="padding: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div class="so-vote-box" style="margin-right: 15px;">
                    <div style="font-size: 22px; font-weight: bold; color: #0C0D0E;">{top_conf:.1f}%</div>
                    <div style="font-size: 11px; color: #6A737C; text-transform: uppercase;">Confidence</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #F48024; text-transform: uppercase;">
                        {top_pred}
                    </div>
                    <div style="font-size: 12px; color: #525960;">
                        Predicted class from 10 CIFAR-10 benchmark visual categories
                    </div>
                </div>
            </div>

            <div style="border-top: 1px solid #E3E6E8; padding-top: 12px;">
                <div style="font-size: 13px; font-weight: 600; color: #232629; margin-bottom: 6px;">Representation Metrics (h &isin; &reals;<sup>512</sup>)</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; font-family: monospace;">
                    <div>&bull; Embedding Dimension: <strong>512-d</strong></div>
                    <div>&bull; L2-Norm: <strong>{l2_norm:.4f}</strong></div>
                    <div>&bull; Embedding Mean: <strong>{feat_mean:.4f}</strong></div>
                    <div>&bull; Embedding Std: <strong>{feat_std:.4f}</strong></div>
                    <div>&bull; Finiteness: <strong style="color: {'#2F6F44' if is_finite else '#D0393E'};">{'VALID' if is_finite else 'NaN/Inf'}</strong></div>
                    <div>&bull; Device: <strong>{DEVICE}</strong></div>
                </div>
            </div>
        </div>
    </div>
    """

    return confidences, stats_html


SAMPLE_DIR = Path("samples")
if not SAMPLE_DIR.is_dir():
    SAMPLE_DIR = Path("app/samples")

examples = []
if SAMPLE_DIR.is_dir():
    examples = [[str(p)] for p in sorted(SAMPLE_DIR.glob("*.png"))]

custom_css = """
/* Stack Overflow Aesthetic Theme */
body, .gradio-container {
    background-color: #F8F9F9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #232629 !important;
}

.so-topbar {
    background-color: #232629;
    border-top: 3px solid #F48024;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    color: #FFFFFF;
    margin-bottom: 20px;
    border-radius: 4px;
}

.so-topbar-title {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

.so-tag {
    display: inline-block;
    padding: 3px 6px;
    font-size: 12px;
    color: #39739D;
    background-color: #E1ECF4;
    border-radius: 3px;
    margin-right: 4px;
    margin-bottom: 4px;
    text-decoration: none;
    font-weight: 500;
}

.so-card {
    background-color: #FFFFFF;
    border: 1px solid #E3E6E8;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}

.so-header-sub {
    background-color: #F8F9F9;
    border-bottom: 1px solid #E3E6E8;
    padding: 10px 15px;
    font-size: 13px;
    font-weight: 600;
    color: #525960;
}

.so-badge-orange {
    background-color: #F48024;
    color: #FFFFFF;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 700;
}

.so-badge-green {
    background-color: #E1F5E6;
    color: #2F6F44;
    border: 1px solid #C3E6CB;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 12px;
    font-weight: 600;
}

.so-vote-box {
    text-align: center;
    padding: 8px 12px;
    border: 1px solid #E3E6E8;
    border-radius: 5px;
    background-color: #FDF7E7;
}

.so-sidebar-box {
    background-color: #FDF7E7;
    border: 1px solid #F1E5BC;
    border-radius: 5px;
    padding: 12px 14px;
    margin-bottom: 15px;
    font-size: 13px;
}

.so-sidebar-title {
    font-weight: 700;
    color: #0C0D0E;
    margin-bottom: 8px;
    font-size: 13px;
    border-bottom: 1px solid #F1E5BC;
    padding-bottom: 4px;
}

.so-btn-orange {
    background-color: #F48024 !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(css=custom_css, title="StackSimCLR — CIFAR-10 Visual Representation Classifier") as demo:
    # Stack Overflow Top Bar
    gr.HTML("""
    <div class="so-topbar">
        <div class="so-topbar-title">
            <span style="color: #F48024; font-size: 22px;">&#9776;</span>
            <span style="color: #FFFFFF;">stack</span><span style="color: #F48024; font-weight: 800;">simclr</span>
        </div>
        <div style="margin-left: 20px; font-size: 13px; color: #BABFC4;">
            Self-Supervised Visual Representation & Classification on CIFAR-10
        </div>
        <div style="margin-left: auto; display: flex; gap: 8px;">
            <span class="so-badge-green">&check; S6 Reproduction: 75.23% Top-1</span>
            <span class="so-badge-green">&check; S7 Scaling Evaluated</span>
        </div>
    </div>
    """)

    # Question Header Block
    gr.HTML("""
    <div style="margin-bottom: 15px;">
        <h1 style="font-size: 24px; font-weight: 600; color: #232629; margin-bottom: 6px;">
            How does SimCLR learn visual representations on CIFAR-10 from scratch without labels?
        </h1>
        <div style="font-size: 12px; color: #6A737C; border-bottom: 1px solid #E3E6E8; padding-bottom: 12px;">
            <span>Asked: <strong>Course Project</strong></span>
            <span style="margin-left: 15px;">Paper: <strong>Chen et al. (ICML 2020)</strong></span>
            <span style="margin-left: 15px;">Reproduction Status: <strong>VERIFIED (Stage S6 & S7 COMPLETE)</strong></span>
            <span style="margin-left: 15px;">Encoder: <strong>ResNet-18 (CIFAR stem)</strong></span>
        </div>
        <div style="margin-top: 8px;">
            <span class="so-tag">pytorch</span>
            <span class="so-tag">simclr-v1</span>
            <span class="so-tag">cifar-10</span>
            <span class="so-tag">contrastive-learning</span>
            <span class="so-tag">resnet-18</span>
            <span class="so-tag">self-supervised</span>
        </div>
    </div>
    """)

    with gr.Row():
        # Main Column: Upload & Prediction
        with gr.Column(scale=3):
            gr.HTML("""
            <div class="so-card">
                <div class="so-header-sub">
                    <strong>1. INPUT IMAGE (TEST BENCHMARK)</strong> &mdash; Upload an image or select a sample
                </div>
            </div>
            """)
            input_image = gr.Image(type="pil", label="Input Image (Auto-resized & normalized)", height=220)

            predict_btn = gr.Button("Classify with SimCLR Representation", variant="primary", elem_classes=["so-btn-orange"])

            output_label = gr.Label(num_top_classes=5, label="Top-5 Classification Confidences")
            output_stats = gr.HTML()

            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    outputs=[output_label, output_stats],
                    fn=predict,
                    run_on_click=True,
                    cache_examples=False,
                    label="CIFAR-10 Test Sample Examples (Click to Test)",
                )

        # Right Column: The Overflow Blog / Project Context
        with gr.Column(scale=2):
            gr.HTML("""
            <div class="so-sidebar-box">
                <div class="so-sidebar-title">The Overflow Blog: SimCLR v1</div>
                <div style="line-height: 1.5; color: #3B4045;">
                    <p style="margin-bottom: 8px;">
                        <strong>What is this app?</strong><br>
                        This system demonstrates <strong>Contrastive Visual Representation Learning</strong>.
                        The base ResNet-18 model was pretrained on unlabeled CIFAR-10 images using the <strong>NT-Xent contrastive loss</strong>.
                    </p>
                    <p style="margin-bottom: 8px;">
                        <strong>Key Architecture:</strong><br>
                        &bull; <strong>Stem:</strong> 3&times;3 conv (stride 1), no max-pool (preserves 32&times;32 resolution).<br>
                        &bull; <strong>MLP Head:</strong> 512 &rarr; 512 &rarr; 128 (discarded after pretrain).<br>
                        &bull; <strong>Representation:</strong> Vector <em>h</em> &isin; &reals;<sup>512</sup> fed to downstream frozen linear probe.
                    </p>
                </div>
            </div>

            <div class="so-sidebar-box" style="background-color: #F2F8FD; border-color: #D3E5F5;">
                <div class="so-sidebar-title" style="border-color: #D3E5F5; color: #1B4F72;">Stage S7: Data Scaling Curve</div>
                <div style="font-size: 12px; line-height: 1.4;">
                    Our data fraction experiment verified monotonic scaling with diminishing returns:
                    <table style="width: 100%; margin-top: 6px; font-size: 11px; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #D3E5F5;">
                            <th style="text-align: left; padding: 2px;">Fraction</th>
                            <th style="text-align: right; padding: 2px;">Samples</th>
                            <th style="text-align: right; padding: 2px;">Top-1 Acc</th>
                        </tr>
                        <tr><td>10%</td><td style="text-align: right;">4,500</td><td style="text-align: right;"><strong>54.17%</strong></td></tr>
                        <tr><td>25%</td><td style="text-align: right;">11,250</td><td style="text-align: right;"><strong>65.37%</strong></td></tr>
                        <tr><td>50%</td><td style="text-align: right;">22,500</td><td style="text-align: right;"><strong>71.04%</strong></td></tr>
                        <tr style="background-color: #E1F5E6;"><td><strong>100%</strong></td><td style="text-align: right;">45,000</td><td style="text-align: right;"><strong>75.23%</strong></td></tr>
                    </table>
                </div>
            </div>

            <div class="so-card">
                <div class="so-header-sub">PROJECT ATTRIBUTION</div>
                <div style="padding: 12px; font-size: 12px; line-height: 1.5; color: #525960;">
                    <div><strong>Course:</strong> Machine Learning Paper Reproduction</div>
                    <div><strong>Institution:</strong> Institute of Business Administration (IBA), Karachi</div>
                    <div style="margin-top: 6px;">
                        <strong>Team Members:</strong><br>
                        &bull; Saalik Hussain Zaidi (36206)<br>
                        &bull; Mubashir Abid (36224)<br>
                        &bull; Muhammad Zohair Khan (33939)
                    </div>
                </div>
            </div>
            """)

    # Predict on button click, and automatically when a new image is uploaded.
    # (Examples run prediction themselves via run_on_click, so no .change handler
    # is needed and nothing fires twice.)
    predict_btn.click(fn=predict, inputs=input_image, outputs=[output_label, output_stats])
    input_image.upload(fn=predict, inputs=input_image, outputs=[output_label, output_stats])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
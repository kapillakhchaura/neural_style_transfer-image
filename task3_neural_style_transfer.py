import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt

# 📁 Image paths on Desktop
CONTENT_IMAGE_PATH = "C:/Users/msi/Desktop/content.jpg"
STYLE_IMAGE_PATH = "C:/Users/msi/Desktop/style.jpg"

# 🖼️ Load and resize image to 256px for speed
def load_image(img_path, max_size=256):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image

# Load content and style images
content = load_image(CONTENT_IMAGE_PATH)
style = load_image(STYLE_IMAGE_PATH)

# Load pre-trained VGG19 model
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# Target image (copy of content)
target = content.clone().requires_grad_(True)

# Loss function and optimizer
mse = nn.MSELoss()
optimizer = optim.Adam([target], lr=0.003)

# Feature layers
style_layers = ['0', '5', '10', '19', '28']
content_layer = '21'

# Feature extraction function
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in style_layers or name == content_layer:
            features[name] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Extract features
style_features = get_features(style, vgg)
content_features = get_features(content, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# 🔁 Style Transfer Loop (only 50 iterations for speed)
for i in range(50):
    target_features = get_features(target, vgg)
    content_loss = mse(target_features[content_layer], content_features[content_layer])
    style_loss = sum(
        mse(gram_matrix(target_features[layer]), style_grams[layer])
        for layer in style_layers
    )
    total_loss = content_loss + 1e6 * style_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {total_loss.item():.4f}")

# 🔽 Convert tensor to image
final_img = target.squeeze().detach().numpy().transpose(1, 2, 0)

# Normalize image pixels to [0,1]
final_img = final_img - final_img.min()
final_img = final_img / final_img.max()

# 💾 Save to Desktop
plt.imsave("C:/Users/msi/Desktop/styled_output.jpg", final_img)
print("✅ Image saved as styled_output.jpg on Desktop.")

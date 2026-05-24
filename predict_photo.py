import torch
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch import nn
from PIL import Image


# DEVICE
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


# MODEL
class AgeGenderModel(nn.Module):

    def _init_(self):

        super()._init_()

        base = efficientnet_b0(weights=None)

        features = base.classifier[1].in_features
        base.classifier = nn.Identity()

        self.backbone = base

        self.age_head = nn.Sequential(
            nn.Linear(features,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(features,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )


    def forward(self,x):

        feat = self.backbone(x)

        age = self.age_head(feat).squeeze(1)
        gender = self.gender_head(feat)

        return age, gender


# LOAD MODEL
model = AgeGenderModel().to(device)

model.load_state_dict(
    torch.load("best_model_v2.pth", map_location=device)
)

model.eval()


# IMAGE TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# LOAD IMAGE
image_path = "test.JPG"

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)


# PREDICT
with torch.no_grad():

    age_pred, gender_pred = model(img_tensor)

    age = age_pred.item() * 100
    gender = "Male" if gender_pred.argmax() == 1 else "Female"


print("\nPrediction:")
print("Age:", int(age))
print("Gender:", gender)


# OPTIONAL: SHOW IMAGE
frame = cv2.imread(image_path)

label = f"{gender}, {int(age)}"

cv2.putText(
    frame,
    label,
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
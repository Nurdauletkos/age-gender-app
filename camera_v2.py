import torch
import cv2
from torchvision import models, transforms
from torch import nn
from PIL import Image




device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)




face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)




class AgeGenderNet(nn.Module):

    def __init__(self):

        super().__init__()

        base = models.resnet18(weights=None)
        base.fc = nn.Identity()

        self.backbone = base

        self.age_head = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,2)
        )


    def forward(self, x):

        feat = self.backbone(x)

        age = self.age_head(feat).squeeze(1)
        gender = self.gender_head(feat)

        return age, gender




model = AgeGenderNet().to(device)

model.load_state_dict(
    torch.load("best_model_v2.pth", map_location=device)
)

model.eval()

print("Model loaded")




transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])




cap = cv2.VideoCapture(0)

print("Camera started (ESC to exit)")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80,80)
    )


    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        inp = transform(pil_img).unsqueeze(0).to(device)


        with torch.no_grad():

            age_pred, gender_pred = model(inp)

            age = age_pred.item() * 100
            gender_id = torch.argmax(gender_pred).item()

            gender = "Male" if gender_id == 1 else "Female"


        label = f"{gender}, {int(age)}"


        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )


    cv2.imshow("Age & Gender (V2)", frame)


    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
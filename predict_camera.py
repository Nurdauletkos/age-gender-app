import torch
import cv2
from torchvision import models, transforms
from torch import nn
from PIL import Image


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Load model
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512,3)
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()


# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


cap = cv2.VideoCapture(0)

print("Press ESC to exit")


while True:

    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )


    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        pil = Image.fromarray(
            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        )

        inp = transform(pil).unsqueeze(0).to(device)


        with torch.no_grad():

            out = model(inp)

            age = out[0,0].item() * 100
            gender_id = torch.argmax(out[0,1:]).item()

            gender = "Male" if gender_id==1 else "Female"


        label = f"{gender}, {age:.0f}"

        cv2.rectangle(
            frame,(x,y),(x+w,y+h),(0,255,0),2
        )

        cv2.putText(
            frame,label,(x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,(0,255,255),2
        )


    cv2.imshow("Age Gender", frame)


    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()

import pygame
import torch
from torchvision import transforms
from PIL import Image

from main import CNN

def run() -> None:
    model = CNN()
    model.load_state_dict(torch.load("mnistmodel.pth"))
    model.eval()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    pygame.init()
    WINDOW_SIZE = 280
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))  # Extra space for prediction text

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    screen.fill(BLACK)

    brush_size = 15
    drawing = False
    prediction = None

    font = pygame.font.SysFont(None, 36)

    def preprocess(surface):
        raw_str = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (WINDOW_SIZE, WINDOW_SIZE), raw_str)
        img = img.convert("L")
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        tensor = transform(img).unsqueeze(0)
        return tensor.to(device)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill(BLACK)
                    prediction = None
                if event.key == pygame.K_p:
                    img_tensor = preprocess(screen.subsurface((0, 0, WINDOW_SIZE, WINDOW_SIZE)))
                    with torch.no_grad():
                        output = model(img_tensor)
                        pred = output.argmax(1).item()
                        prediction = pred
        if drawing:
            mx, my = pygame.mouse.get_pos()
            if my < WINDOW_SIZE:
                pygame.draw.circle(screen, WHITE, (mx, my), brush_size)

        screen.fill(BLACK, (0, WINDOW_SIZE, WINDOW_SIZE, 40))
        if prediction is not None:
            text = font.render(f"Prediction: {prediction}", True, WHITE)
            screen.blit(text, (10, WINDOW_SIZE + 5))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run()
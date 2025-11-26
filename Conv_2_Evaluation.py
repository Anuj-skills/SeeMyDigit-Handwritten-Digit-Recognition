from Conv_2_Training import *

#Evaluation 
model.eval()  

correct = 0
total = 0

with torch.no_grad():  
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save Model
torch.save(model.state_dict(), "model_2.pth")
print("Model Saved Successfully!")

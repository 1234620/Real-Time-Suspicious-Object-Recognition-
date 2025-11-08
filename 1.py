import os
from ultralytics import YOLO
import cv2
from pathlib import Path

def train_model(data_yaml, epochs=50):
    """Train the YOLOv8 model"""
    # Load a model
    model = YOLO('yolov8n.pt')  # load pretrained model
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='my_model'
    )
    
    return model

def detect_objects(model_path=None):
    """Run real-time object detection"""
    # Load the model
    if model_path:
        model = YOLO(model_path)  # custom trained model
    else:
        model = YOLO('yolov8n.pt')  # pretrained model
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    print("Starting detection... Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame, conf=0.5)  # confidence threshold 0.5
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("Object Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Object Detection System")
    print("----------------------")
    
    while True:
        print("\nChoose an option:")
        print("1. Train new model (requires dataset)")
        print("2. Run detection with pretrained model")
        print("3. Run detection with custom trained model")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            # Train new model
            data_yaml = input("Enter path to your data.yaml file: ")
            if not os.path.exists(data_yaml):
                print("Error: data.yaml file not found")
                continue
                
            epochs = input("Enter number of epochs (default 50): ")
            epochs = int(epochs) if epochs else 50
            
            print("\nStarting training...")
            model = train_model(data_yaml, epochs)
            print("Training completed!")
            
        elif choice == '2':
            # Run detection with pretrained model
            print("\nStarting detection with pretrained model...")
            detect_objects()
            
        elif choice == '3':
            # Run detection with custom model
            model_path = input("Enter path to your trained model (.pt file): ")
            if not os.path.exists(model_path):
                print("Error: Model file not found")
                continue
                
            print("\nStarting detection with custom model...")
            detect_objects(model_path)
            
        elif choice == '4':
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
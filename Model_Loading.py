from tensorflow.keras.models import load_model

# Load the model
model = load_model("C:/Users/Asus/Desktop/Project/Agri_World/Agri_World/models/DL_models/strawberry_model.h5")

# Print the model summary to inspect its architecture
model.summary()

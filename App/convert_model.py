import keras  # Explicitly import keras first
from keras.models import model_from_json
from pathlib import Path

# Paths to model files
json_path = Path("D:/capstone/interview-score-prediction/App/Models/Personality_traits_NN.json")
weights_path = Path("D:/capstone/interview-score-prediction/App/Models/Personality_traits_NN.h5")
output_path = Path("D:/capstone/interview-score-prediction/App/Models/Personality_traits_NN_full.h5")

try:
    # Load JSON model architecture
    with open(json_path, "r") as json_file:
        model = model_from_json(json_file.read())

    # Load weights
    model.load_weights(weights_path)

    # Compile model (optional, ensures compatibility)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Save as full HDF5 model
    model.save(output_path)
    print(f"Model successfully saved to {output_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Conversion failed: {e}")


---

# Text Generation with LSTM

This project demonstrates how to train an LSTM-based language model for text generation using the OpenWebText dataset. The model is trained on a large corpus of text and can generate coherent text based on a given prompt.

## Features
- **Data Downloading and Extraction**: Automatically downloads and extracts the OpenWebText dataset.
- **Tokenizer Training**: Trains a Byte-Pair Encoding (BPE) tokenizer for text preprocessing.
- **LSTM Model**: Implements an LSTM-based neural network for text generation.
- **Text Generation**: Generates text using temperature sampling for controlled randomness.

## Requirements
- Python 3.7 or higher
- PyTorch
- Tokenizers library
- Requests library
- Pylzma library
- Tqdm library

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-generation-lstm.git
   cd text-generation-lstm
   ```

2. Install the required Python libraries:
   ```bash
   pip install torch tokenizers requests pylzma tqdm
   ```

   If you have a GPU and want to use it, install the GPU version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Replace `cu118` with the appropriate CUDA version for your system.

## Usage
1. **Run the Script**:
   Execute the script to train the model and generate text:
   ```bash
   python text_generation.py
   ```

2. **Monitor Progress**:
   - The script will download and extract the dataset (if not already downloaded).
   - It will train or load a tokenizer.
   - It will prepare the dataset and train the LSTM model (if no pre-trained model is found).
   - Finally, it will generate text using the trained model.

3. **Generated Text**:
   The script will generate text starting with the prompt `"Once upon a time"`. The output will be displayed in the terminal.

## Customization
You can modify the following parameters in the script to customize the behavior:
- **Start Text**: Change the `start_text` variable to use a different prompt for text generation.
- **Hyperparameters**: Adjust `VOCAB_SIZE`, `EMBEDDING_DIM`, `HIDDEN_DIM`, `NUM_LAYERS`, `BATCH_SIZE`, `SEQ_LENGTH`, `LEARNING_RATE`, and `GRAD_CLIP` for model tuning.
- **Temperature**: Modify the `temperature` parameter in the `generate_text` function for more or less randomness in text generation.

## Example Output
After running the script, you should see output similar to the following:
```
2023-10-15 12:34:56,123 - INFO - Data extraction complete.
2023-10-15 12:35:10,456 - INFO - Tokenizer training complete.
2023-10-15 12:35:15,789 - INFO - Epoch 1/5: Loss = 4.567
2023-10-15 12:35:20,123 - INFO - Epoch 2/5: Loss = 3.456
...
2023-10-15 12:36:00,456 - INFO - Model saved successfully!

Generated Text:
Once upon a time, there was a little girl who lived in a small village. She loved to explore the forest and discover new things. One day, she found a magical stone that granted her the power to talk to animals...
```

## Troubleshooting
1. **Out of Memory (OOM) Errors**:
   - Reduce `BATCH_SIZE` or `SEQ_LENGTH`.
   - Use a smaller dataset or model.

2. **Tokenizer Errors**:
   - Ensure the `tokenizers` library is installed correctly.
   - Check if the dataset file (`openwebtext.txt`) is properly extracted.

3. **CUDA Errors**:
   - Ensure you have the correct version of PyTorch installed for your GPU.
   - If no GPU is available, the script will fall back to CPU.

4. **Network Errors**:
   - If the dataset download fails, check your internet connection or manually download the dataset from the provided URL.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### How to Use the README
1. Save the content above as `README.md` in your project directory.
2. Update the `your-username` and repository URL in the "Installation" section to match your GitHub profile and repository.
3. Add a `LICENSE` file if you want to include licensing information.


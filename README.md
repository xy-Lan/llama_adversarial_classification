# Adversarial Classification Project

This project explores adversarial attacks on various language models fine-tuned for fact-checking tasks.

## Code Structure and Model Scripts

This repository contains scripts for fine-tuning models (Phase A), applying a secondary fine-tuning phase for adversarial robustness (Phase B), and classifying samples from the CheckThat! Lab dataset using the fine-tuned models.

### Model Classification Scripts (CheckThat! Lab Dataset)

Scripts used to classify samples from the CheckThat! Lab dataset with models fine-tuned in Phase A or Phase B:

*   **Llama Models**: `scripts/classify_adversarial.py`
*   **Mistral Models**: `scripts/classify_adversarial_mistral.py`
*   **Qwen Models**: `scripts/classify_adversarial_qwen.py`

### Phase A: Initial Fine-tuning (FEVER Dataset)

Scripts for the initial fine-tuning of language models on the FEVER dataset:

*   **Llama (3B)**: `scripts/phaseA_llama3b_fever.py`
*   **Mistral (7B)**: `scripts/phaseA_mistral7b_fever.py`
*   **Qwen**: `scripts/finetune_qwen_fever.py`

### Phase B: Adversarial Fine-tuning (Paired Samples)

Scripts for the secondary fine-tuning phase, focusing on adversarial robustness using paired original and adversarial samples:

*   **Llama, Mistral, and Qwen Models**: `scripts/phaseB_pairs_only.py`
    *   The specific model to be fine-tuned in this phase is determined by script arguments (e.g., by providing the path to a Phase A fine-tuned model).

Once you have completed the fine-tuning process using scripts/phaseB_pairs_only.py, you can evaluate the performance of your Phase B models on adversarial classification tasks.

Please use the corresponding classification script below based on the model you fine-tuned (Llama, Mistral, or Qwen).
*   **Llama Models**: `scripts/classify_adversarial.py`
*   **Mistral Models**: `scripts/classify_adversarial_mistral.py`
*   **Qwen Models**: `scripts/classify_adversarial_qwen.py`

Note: Replace the default test data set with the test set path scripts_csv/test.csv

You can specify the test dataset path using the --data_file argument, for example:
python scripts/classify_adversarial.py ----data_file scripts_csv/test.csv
## Usage

Please refer to the individual scripts for specific command-line arguments and usage instructions. Ensure that the necessary datasets (FEVER, CheckThat! Lab, adversarial samples) are correctly placed and paths are configured within the scripts or via arguments.

You may need to load appropriate environment modules for CUDA and Python, for example:
```bash
module load CUDA/11.8.0
module load Python/3.10.4-GCCcore-11.3.0
```

Then, run the desired Python script:
```bash
python scripts/your_chosen_script.py --your_arguments
``` 

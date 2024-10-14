# SAM_Molmo_Whisper
A simple integration of Segment Anything Model, Molmo, and, Whisper to segment objects using voice and natural language.

Capabilities:

* Segment objects with **SAM2.1** using point prompts.
* Points can be obtained by **prompting Molmo** with natural language. Molmo can take inputs by the **text box (typing)** or **Whisper via microphone (speech to text)**.

**Run the Gradio demo using**:

```
python app.py
```

https://github.com/user-attachments/assets/66a0620e-ede3-4018-8ee7-f261790747cb

## Installing Requirements

Install Pytorch, Hugging Face Transformers, and rest of the base requirements.

```
pip install -r requirements.txt
```

**Install SAM2:**

*It is highly recommended to clone SAM2 a separate directory other than this project directory and run the installation commands*.

```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```


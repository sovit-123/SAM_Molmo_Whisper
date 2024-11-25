# SAM_Molmo_Whisper

***Note: The project is in very initial stages and will change drastically in the near future. Things may break.***

**[Go to Setup](#Setup)**

A simple integration of Segment Anything Model, Molmo, and, Whisper to segment objects using voice and natural language.

Capabilities:

* Segment objects with **SAM2.1** using point prompts.
* Points can be obtained by **prompting Molmo** with natural language. Molmo can take inputs by the **text box (typing)** or **Whisper via microphone (speech to text)**.

**Run the Gradio demo using**:

```
python app.py
```

https://github.com/user-attachments/assets/66a0620e-ede3-4018-8ee7-f261790747cb

## What's New

### October 30, 2024

* Added tabbed interface for video segmentation. Process remains the same. Either prompt via text or voice, upload a video and get the segmentation maps of the objects.

## Setup

### Clone Repo

```
git clone https://github.com/sovit-123/SAM_Molmo_Whisper.git
```

```
cd SAM_Molmo_Whisper
```

### Installing Requirements

Install Pytorch, Hugging Face Transformers, and the rest of the base requirements.

```
pip install -r requirements.txt
```

### Install SAM2

*It is highly recommended to clone SAM2 to a separate directory other than this project directory and run the installation commands*.

```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

## To Use CLIP Auto Labelling

After installing the requirements install SpaCy's  `en_core_web_sm` model.

```
spacy download en_core_web_sm
```

### Run the App

```
python app.py
```


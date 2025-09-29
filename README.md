# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts
[![arXiv](https://img.shields.io/badge/arXiv-2509.04844-b31b1b.svg)](https://www.arxiv.org/abs/2509.04844)

The source code for **REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts**.

# 🔥 News

🎉🎉🎉 **[July. 2025]** We are delighted to announce that our paper, **"REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts"**, has been accepted by ACM MM 2025!

</h5>
<p align="center">
    <img src="./Image/model.jpg" alt="Pipeline" style="width:75%; height:auto;">
</p>

📆 **[July. 2025]** [UMRE dataset](https://drive.google.com/file/d/17N_GVv5sCnT55ZUi_5dXl66bac1TuUlC/view?usp=sharing) has been released. Prompt and Explanations for various relations can be found in ACM_supplement.pdf.

# 🏆 UMRE Dataset
</div>

<br>

<div align="center">
<img src='./Image/dataset.jpg' width='60%'>
</div>

Our UMRE dataset is a further development of the [MNRE dataset](https://github.com/thecharm/MNRE) and [MORE dataset](https://github.com/NJUNLP/MORE).

---

## 📊 Further Experience

### Question 1: The benefits of MLLM-generated captions
As shown in Fig.5 of our paper, we have compared results across different captions and without captions. Captions are effective in bridging the modality gap between image and text . Additionally, we invite graduate students with strong English proficiency to evaluated caption accuracy (CapAcc) and length (CapLen). Results are as follows:

| MLLM           | CapLen | CapAcc (\%) | F1-Score (\%) |
|----------------|--------|-------------|---------------|
| Without Caption|        |             | 63.11         |
| BLIP2          | 12.02  | 83.15       | 66.36         |
| Llama3.2-11B   | 40.36  | 89.27       | 67.57         |
| Qwen2-VL-7B    | 32.76  | 91.31       | 68.23         |
| Qwen2.5-VL-7B  | 28.43  | 93.37       | **69.17**         |

The table shows that caption length does not directly correlate with performance; instead, caption accuracy (CapAcc) is the key factor affecting results .

### Question 2: Annotation Consistency of the UMRE Dataset
Because ambiguous hierarchical relations exist, such as "/per/org/member of" and "/per/org/leader of", there may be discrepancies in annotation results among different annotators. To resolve such conflicts, a third independent adjudicator was introduced to evaluate and finalize the annotations. Per-Modality and Per-Relation Kappa Value as follows:

### Table 1: Kappa Value by Per Relational Triplet Type
| Relational Triplet Type                | Kappa Value   |
|----------------------------------------|---------------|
| textual entities-textual entities      | 0.6831        |
| visual objects-visual objects          | 0.6748        |
| textual entities-visual objects        | 0.8173        |

### Table 2: Kappa Value for Per Relation Type
| Relation Type                          | Kappa Value |
|--------------------------------------|---------------|
| none                                 | 0.7273        |
| /per/loc/place_of_governance         | 0.6956        |
| /per/misc/party                      | 0.8613        |
| /per/org/member_of                   | 0.6656        |
| /per/per/self                        | 0.9233        |
| /per/misc/nationality                | 0.7192        |
| /loc/loc/self                        | 0.9280        |
| /per/misc/present_in                 | 0.6873        |
| /per/loc/place_of_residence          | 0.6543        |
| /org/org/self                        | 0.8335        |
| /misc/misc/self                      | 0.8700        |
| /per/per/opponent                    | 0.6219        |
| /per/loc/place_of_birth              | 0.6758        |
| /per/per/partner                     | 0.6569        |
| /per/org/opposed_to                  | 0.6712        |
| /loc/loc/contain                     | 0.8390        |
| /org/loc/locate_at                   | 0.6380        |
| /per/misc/president                  | 0.8123        |
| /misc/loc/held_on                    | 0.6649        |
| /per/org/leader_of                   | 0.6077        |
| /org/org/subsidiary                  | 0.7072        |
| /per/per/relatives                   | 0.8889        |
| /per/misc/awarded                    | 0.6707        |
| /misc/misc/part_of                   | 0.6913        |
| /per/misc/race                       | 0.6955        |
| /per/per/alumni                      | 0.6666        |
| /per/misc/religion                   | 0.6315        |
| /org/misc/present_in                 | 1.0000        |

Kappa values are used to measure the agreement between annotators.

---

## 📦 Installation Guide

### 1.  Download Required Datasets

#### UMRE Dataset
Download the [UMRE Dataset](https://drive.google.com/file/d/17N_GVv5sCnT55ZUi_5dXl66bac1TuUlC/view?usp=sharing) and extract it:
```bash
unzip UMRE_Data.zip -d datasets/
```

#### UMKE Partner Supplementary Files
Download the [UMKE Partner Supplementary Files](https://drive.google.com/file/d/1ozJ25WaSnHJ7De84tdAWncBU9YV57nYG/view?usp=sharing) and extract them:
```bash
unzip umke_partner.zip -d datasets/
```

### 2.  Generate Depth Maps

#### Step 1: Set up Depth Estimation Model
1.  Clone the official [Depth-Anything-V2 repository](https://github.com/DepthAnything/Depth-Anything-V2):
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
```
2.  Follow their installation instructions to set up dependencies.

#### Step 2: Process UMKE Images
Return to your ROMOTE project directory and generate depth maps:
```bash
python ROMOTE_code/depth_data/test.py 
```
This will generate corresponding depth maps for all images in the UMKE dataset.

---

### 3.  Execute Training/Inference

Run the main pipeline with optimized configurations:
```bash
bash ROMOTE_code/run_umke_best.sh
```

> **Tip**: Ensure all dependencies are installed and paths in `run_umke_best.sh` match your directory structure.

---

# Acknowledgement  
Our dataset is extended based on the methods from [RIVEG](https://github.com/JinYuanLi0012/RiVEG) and [PGIM](https://github.com/JinYuanLi0012/PGIM) on the [MNRE dataset](https://github.com/thecharm/MNRE) and [MORE dataset](https://github.com/NJUNLP/MORE), followed by manual annotation and correction of relational triplets.  

Our code is built upon the open-sourced [HVFormer](https://github.com/liuxiyang641/HVFormer) and [MOREformer](https://github.com/NJUNLP/MORE). Thanks for their great work!


# **Glioblastoma Progression Modeling with Hidden Markov Models (HMMs)**

This project applies **Hidden Markov Models (HMMs)** to model **glioblastoma (GBM) tumor progression** using longitudinal MRI-based features and clinical assessments. The goal is to infer **hidden tumor states**—such as stable disease, pseudoprogression, and true progression—and identify early transition into aggressive disease.

Glioblastoma progression is not directly observable at the biological level, but imaging and clinical metrics contain **noisy signals** that reflect underlying behavior. An HMM captures these hidden dynamics and reconstructs the most likely disease trajectory using the **Viterbi algorithm**.

---

## **🧠 Project Overview**

Glioblastoma patients undergo multiple MRI scans and clinical evaluations over time. For each timepoint, we extract or use provided features such as:

- Tumor subregion volumes (enhancing core, whole tumor, edema)
- Quantitative radiomic features
- RANO-style clinical assessments (stable, partial response, progression)
- Optional: changes in volume (Δ volume), growth rate, treatment phase, survival outcome

The HMM models three latent tumor states:

- **State 1 — Stable / Responding**
- **State 2 — Indeterminate / Possible Pseudoprogression**
- **State 3 — True Progression / Relapse**

The model learns:

1. **Transition probabilities** between tumor states  
2. **Emission probabilities** linking MRI features to hidden tumor severity  
3. **Most likely tumor-state sequence** across each patient’s follow-up timeline  

This enables early identification of high-risk transitions into true progression.

---

## **📊 Key Features**

- End-to-end pipeline for glioblastoma progression modeling  
- Uses public GBM datasets (BraTS, TCGA-GBM, or longitudinal MRI datasets)  
- Extracts tumor volumes and clinical features for each timepoint  
- Implements **Baum–Welch** for model training  
- Applies **Viterbi decoding** to infer hidden disease states  
- Generates patient-level tumor-state trajectories  
- Evaluates transition dynamics and progression patterns  
- Interpretation-focused, clinically aligned modeling

---

## **📁 Dataset Sources**

This project supports multiple publicly available glioblastoma datasets:

- **BraTS (Brain Tumor Segmentation Challenge)** – multimodal MRI with tumor subregion labels  
- **Longitudinal GBM MRI datasets** with follow-up timepoints and RANO assessments  
- **TCGA-GBM clinical + radiomic features**

All datasets are fully de-identified and publicly accessible.

---

## **⚙️ Methods**

- **Hidden Markov Model (HMM)** with 3 latent tumor states  
- **Gaussian emissions** from MRI-derived tumor features  
- **Baum–Welch algorithm** for parameter learning  
- **Viterbi algorithm** for tumor-state sequence decoding  
- **Temporal radiomics** derived from longitudinal MRI  
- **Plots of inferred hidden states over time**  
- **Transition matrix analysis** to examine progression risk

---

## **📈 Example Outputs**

- Reconstructed tumor-state trajectory for each patient  
- Transition matrix showing likelihood of Stable → Progression  
- Plots of tumor volume vs. inferred hidden state  
- Detection of early transition into aggressive relapse  
- Distinction between pseudoprogression and true progression  

---

## **🚀 How to Run**

```bash
pip install -r requirements.txt
python preprocess.py
python train_hmm.py
python infer_states.py
python visualize_states.py
```

You may configure the number of states, emission type, or feature set in `config.json`.

---

## **🔍 Project Goals**

- Build an interpretable, probabilistic model of glioblastoma progression  
- Detect early transition into true progression  
- Integrate radiomics, MRI sequences, and clinical assessments  
- Demonstrate real-world application of sequence modeling in neuro-oncology  
- Provide an explainable alternative to deep learning for longitudinal tumor analysis

---

## **📚 Background**

Glioblastoma is a highly aggressive brain tumor characterized by unpredictable growth and treatment response. Traditional imaging markers can be ambiguous (e.g., pseudoprogression vs. true progression).  
HMMs provide:

- A **structured, interpretable** approach  
- Ability to model latent biological states  
- A probabilistic view of patient trajectories  
- Clinically meaningful state transitions  

This makes them well-suited for understanding tumor evolution.

---

## **📝 Status**

✔ Data preprocessing pipeline  
✔ Feature extraction from MRI or radiomic tables  
✔ HMM implementation  
✔ Viterbi decoding of tumor-state trajectories  
✔ Visualizations and transition matrices  
⬜ Add survival analysis by inferred state  
⬜ Expand to 4–5 latent states  
⬜ Explore hybrid HMM-LSTM approach

---

## **📄 License**

MIT License.

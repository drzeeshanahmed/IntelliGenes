# IntelliGenes

This is the CLI implementation of _IntelliGenes_. A GUI version is available in the intelligenes-gui branch. 

IntelliGenes is a Python-based portable pipeline that addresses challenges arising from the cascading volume of genomics datasets being created that require interpretation. IntelliGenes serves as a comprehensive toolkit, fitting cutting-edge algorithms for discovering disease-associated biomarkers and patient prediction to users’ unique cohorts. IntelliGenes integrates demographics with genomics, facilitating investigations that consider both variables simultaneously. With IntelliGenes, we introduce I-Genes Scores, our novel metric for understanding the relevance of biomarkers in disease prediction engines.

_IntelliGenes_ can be installed through our GitHub using the terminal. Follow the provided steps to install IntelliGenes and the package’s dependencies: 
```
# Clone IntelliGenes’ GitHub Repository
git clone https://github.com/drzeeshanahmed/intelligenes.git

# Navigate to IntelliGenes
cd intelligenes/

# Install IntelliGenes
pip install .
```

_IntelliGenes_ offers a robust selection of tools to help users understand their multi-genomics datasets. _IntelliGenes_ has been designed as an easy-to-understand pipeline for those at all levels of computational understanding. _IntelliGenes_ has three functions:
```
# Discover Biomarkers
igenes_select -i data/cigt_file.csv -o results/

# Disease Prediction & I-Genes Scores 
igenes_predict -i data/cigt_file.csv -f features_file.csv -o results/

# IntelliGenes (Discovering Biomarkers & Predicting Disease) 
igenes -i data/cigt_file.csv -o results/
```

These are sample commands. We have provided an example CIGT file in tests/.

These commands all users to write various flags that will tailor _IntelliGenes_ to their exact needs: 
```
# IntelliGenes Selection Help
igenes_select --help

# IntelliGenes Prediction Help
igenes_predict --help

# IntelliGenes Help
igenes --help
```

_IntelliGenes_ requires a CIGT formatted dataset as an input. Examples of CIGT datasets can be found on our GitHub. The CIGT formatted dataset integrates demographics and transcriptomic: 
  -	Columns contain demographic or transcriptomic biomarkers, while rows contain identifiers for individual patients. 
  -	Demographics such as ‘Age’, ‘Race’, and ‘Sex’ should be integers (use EHR standards). These demographics are not required, as IntelliGenes works using only genomics/transcriptomics.
  -	There must be a ‘Type’ column, denoting a patient’s status as an integer (use 0 or 1). 

More information is available in **Supplementary Material 2: _IntelliGenes_: Installation, configuration, and user’s guidelines**

If using _IntelliGenes_, please cite: 

Degroat, W., Mendhe, D., Bhurasi, A., Abdelhalim, H., Saman, Z., & Ahmed, Z. (2023). IntelliGenes: A novel machine learning pipeline for biomarker discovery and predictive analysis using multi-genomic profiles. Bioinformatics. 39, 12. btad755. PMID: 38096588. doi:10.1093/bioinformatics/btad755 (Oxford).

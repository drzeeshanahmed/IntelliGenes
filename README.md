# IntelliGenes

IntelliGenes can be installed through our GitHub using the terminal. Follow the provided steps to install IntelliGenes and the package’s dependencies: 
```
# Clone IntelliGenes’ GitHub Repository
git clone https://github.com/drzeeshanahmed/intelligenes.git

# Navigate to IntelliGenes
cd intelligenes/

# Install IntelliGenes
pip install .
```

IntelliGenes offers a robust selection of tools to help users understand their multi-genomics datasets. IntelliGenes has been designed as an easy-to-understand pipeline for those at all levels of computational understanding. IntelliGenes has three functions:
```
# Discover Biomarkers
igenes_select -i /data/cgit_file.csv -o results/

# Disease Prediction & I-Genes Scores 
igenes_predict -i /data/cgit_file.csv -f features_file.csv -o results/

# IntelliGenes (Discovering Biomarkers & Predicting Disease) 
igenes -I /data/cgit_file.csv -o results/
```

These commands all users to write various flags that will tailor IntelliGenes to their exact needs: 
```
# IntelliGenes Selection Help
Igenes_select --help

# IntelliGenes Prediction Help
Igenes_predict --help

# IntelliGenes Help
igenes --help
```

IntelliGenes requires a CGIT formatted dataset as an input. Examples of CGIT datasets can be found on our GitHub. The CGIT formatted dataset integrates demographics and transcriptomic: 
  -	Columns contain demographic or transcriptomic biomarkers, while rows contain identifiers for individual patients. 
  -	Demographics such as ‘Age’, ‘Race’, and ‘Sex’ should be integers (use EHR standards). These demographics are not required, as IntelliGenes works using only genomics/transcriptomics.
  -	There must be a ‘Type’ column, denoting a patient’s status as an integer (use 0 or 1). 

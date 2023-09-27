import argparse
import subprocess
import pkg_resources

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cgit_file', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--n_splits', type=int, default=5)
    
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--no_rfe', action='store_true')
    parser.add_argument('--no_pearson', action='store_true')
    parser.add_argument('--no_chi2', action='store_true')
    parser.add_argument('--no_anova', action='store_true')
    
    parser.add_argument('--voting', type=str, default='soft')
    parser.add_argument('--no_rf', action='store_true')
    parser.add_argument('--no_svm', action='store_true')
    parser.add_argument('--no_xgb', action='store_true')
    parser.add_argument('--no_knn', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--no_igenes', action='store_true')
    parser.add_argument('--no_visualization', action='store_true')
    
    args = parser.parse_args()

    selection_path = pkg_resources.resource_filename('intelligenes', 'selection.py')
    selection_args = [
        'python', selection_path,
        '-i', args.cgit_file,
        '-o', args.output_dir,
        '--random_state', str(args.random_state),
        '--test_size', str(args.test_size),
    ]
    
    if args.normalize:
        selection_args.append('--normalize')
    if args.no_rfe:
        selection_args.append('--no_rfe')
    if args.no_pearson:
        selection_args.append('--no_pearson')
    if args.no_chi2:
        selection_args.append('--no_chi2')
    if args.no_anova:
        selection_args.append('--no_anova')
    
    selection = subprocess.run(selection_args, capture_output=True, text=True)
    print(selection.stdout)
    selection_output = selection.stdout.strip().split('\n')
    for line in selection_output:
        if line.startswith(" Selected Features:"):
            features_file = line.split(":")[1].strip()
            break

    classification_path = pkg_resources.resource_filename('intelligenes', 'classification.py')
    classification_args = [
        'python', classification_path,
        '-i', args.cgit_file,
        '-f',features_file,
        '-o', args.output_dir,
        '--random_state', str(args.random_state),
        '--test_size', str(args.test_size),
        '--n_splits', str(args.n_splits),
        '--voting', args.voting,
    ]
    
    if args.no_rf:
        classification_args.append('--no_rf')
    if args.no_svm:
        classification_args.append('--no_svm')
    if args.no_xgb:
        classification_args.append('--no_xgb')
    if args.no_knn:
        classification_args.append('--no_knn')
    if args.no_mlp:
        classification_args.append('--no_mlp')
    if args.tune:
        classification_args.append('--tune')
    if args.normalize:
        classification_args.append('--normalize')
    if args.no_igenes:
        classification_args.append('--no_igenes')
    if args.no_visualization:
        classification_args.append('--no_visualization')
        
    subprocess.check_call(classification_args)

if __name__ == '__main__':
    main()

# MACE: Multi-Annotator Competence Estimation

Ask 5 people to label or rate something, and you likely get several different answers. But for ML (and lots of other applications), you usually need a single aggregated answer. Using the majority vote is easy… but often wrong. However, disagreement isn’t noise–it’s information. It can mean the item is genuinely hard, or that someone wasn’t paying attention. 

MACE is an Expectation-Maximization (EM)-based algorithm that uses variational inference with Bayesian priors to simultaneously:
- Learn the most likely aggregate labels for items from multiple annotators
- Estimate the competence (reliability) of each annotator
- Model how difficult each item is

It models annotators as either "knowing" the correct answer or "guessing" according to some strategy. 

## Features

- ✅ Supports **discrete categorical labels** (default) and **continuous numeric values**
- ✅ Can incorporate **control items** (known ground truth) for semi-supervised learning
- ✅ Allows specifying **label priors** (if known)
- ✅ Provides **confidence estimates** via entropy calculations
- ✅ Optional **distribution output** shows full probability distributions
- ✅ Handles **missing annotations** (empty cells in CSV)

## Installation

### Requirements

- Python 3.6 or higher
- NumPy
- SciPy

### Install Dependencies

```bash
pip install numpy scipy
```

## Usage

### Basic Command

```bash
python3 mace.py [options] <CSV input file>
```

### Options

| Option | Description |
|--------|-------------|
| `--help` | Display help information |
| `--version` | Display version information |
| `--alpha <FLOAT>` | First hyperparameter of beta prior for Variational Bayes EM (default method). alpha > beta means we assume most annotators are unreliable. Default: 0.5 |
| `--beta <FLOAT>` | Second hyperparameter of beta prior for Variational Bayes EM (default method). beta > alpha means we assume most annotators are reliable. Default: 0.5 |
| `--continuous` | Interpret data values as continuous numeric (returns weighted averages weighted by competence) |
| `--controls <FILE>` | File with control items (i.e., known ground truth labels) for semi-supervised learning. Each line corresponds to one item, so the number of lines MUST match the input CSV file. Control items usually improve accuracy. |
| `--distribution` | Output full probability distributions instead of single predictions in '[prefix.]prediction' |
| `--em` | Use regular EM (Maximum Likelihood Estimation) instead of Variational Bayes EM (default). Performance is usually worse than Variational. |
| `--entropies` | Write entropy values (uncertainty measure) for each item to a separate file '[prefix.]entropies' |
| `--headers` | Add header rows to output files describing column contents |
| `--iterations <INT>` | Number of EM iterations per restart (1-1000). Default: 50 |
| `--prefix <STRING>` | Prefix for output files (e.g., `out` → `out.prediction`) |
| `--priors <FILE>` | File with label priors (tab-separated "label\\tweight" pairs). All labels in the data must be covered. Weights will be normalized to probabilities |
| `--restarts <N>` | Number of random restarts (1-1000). More restarts can find better solutions. Default: 10 |
| `--smoothing <FLOAT>` | Smoothing parameter added to fractional counts for regular EM. Default: 0.01/num_labels |
| `--test <FILE>` | Test file with gold standard labels for evaluation (reports accuracy or RMSE). Each line corresponds to one item in the CSV file, so the number of lines must match. |
| `--threshold <FLOAT>` | Entropy threshold (0.0-1.0). Filter out uncertain instances by returning only the top n%. Default: 1.0 |

## Input Files

### 1. Input Format

The main input file with the annotations must be a **comma-separated (CSV) file** where:
- Each **row** represents one instance/item  (rows can be empty, for example to separate input blocks, and to enable sequence labeling/time step prediction).
- Each **column** represents one annotator (**empty cells** indicate missing annotations)
- File should be formatted in **UTF-8** to avoid problems with newline characters

**Note**: Make sure the last line has a line break.

### Example Input (Discrete Labels with 5 Annotators and Empty Line)

```
NOUN,,,NOUN,PRON
VERB,VERB,,VERB,

ADJ,,ADJ,,ADV
,VERB,,VERB,ADV
NOUN,,,NOUN,PRON
```

#### Example Input (Continuous Values)

```
3.5,4.2,,,3.8,3.9,4.1
,,4.0,4.5,,3.7,3.6,4.3
4.1,3.9,3.8,4.2,,4.0,,3.7
```

### 2. Label Priors
By default, MACE uses a uniform prior over labels (1/num_labels for each label). The **prior** file is optional, and gives the a-priori prevalence of the individual labels (if we know them). We can supply this to MACE with `--priors <FILE>`. The file needs to list all labels (one per line) and tab-separated the weight, probability, or frequency (MACE automatically normalizes these).

- **Format**: Tab-separated "label\\tweight" pairs, one per line
- **Normalization**: Weights are automatically normalized to sum to 1.0
- **Validation**: All labels in the data must be present in the priors file
- **Usage**: Priors are used in the E-step to compute gold label marginals

#### Example Input (Discrete Labels)

```
NOUN	30
VERB	30
ADJ	20
ADV	10
PRON	10
```

### 3. Control Items

If we know the correct answer for some items, we can include **control items** via `--controls <FILE>`. This helps MACE assess annotator reliability in semi-supervised learning. The file with control items needs to have the same number of lines as the input file, with the correct labels specified for the control items.

#### Example Input (Discrete Labels):
```
PRON





NOUN
```


### 4. Test File

If we know *all* answers and only want to get the performance for MACE, we can supply a **test file** via `--test <FILE>`. This file must have the same number of lines as the input file. MACE will output an accuracy score.

#### Example Input (Discrete Labels)

```
PRON
VERB

ADJ
VERB
NOUN
```


## Output Files

MACE generates the following output files:

### 1. Predictions (`<prefix>.prediction`)

- **Discrete mode**: One label per line (most likely label for each instance)
- **Continuous mode**: One weighted average per line
- **Distribution mode**: Tab-separated distributions (see `--distribution` option)
- Empty lines indicate instances filtered by threshold or with no input annotations

This file has the same number of lines as the input file. 

#### Example Output 

```
NOUN
VERB

ADJ
VERB
NOUN
```

If you set --distribution, each line contains the distribution over answer values, sorted by entropy. 

#### Example Output

```
NOUN 0.9997443833265887	PRON 7.140381903855615E-5	ADJ 6.140428479093134E-5	VERB 6.140428479093134E-5	ADV 6.140428479093134E-5
VERB 0.9999961943848287	NOUN 9.514037928812883E-7	ADJ 9.514037928812883E-7	PRON 9.514037928812883E-7	ADV 9.514037928812883E-7

ADJ 0.9990184050335877	ADV 2.741982824057974E-4	NOUN 2.3579889466878394E-4	VERB 2.3579889466878394E-4	PRON 2.3579889466878394E-4
VERB 0.9994950838119411	ADV 1.4104305366466138E-4	NOUN 1.2129104479807625E-4	ADJ 1.2129104479807625E-4	PRON 1.2129104479807625E-4
NOUN 0.9997443833265887	PRON 7.140381903855615E-5	ADJ 6.140428479093134E-5	VERB 6.140428479093134E-5	ADV 6.140428479093134E-5
```

### 2. Competence Scores (`<prefix>.competence`)

- One line with tab-separated values
- Each value (0-1) represents the reliability of one annotator
- Higher values = more reliable annotator

* the **competence estimate** for each annotator, `[prefix.]competence`. This file has one line with tab separated values. In the POS example from above, this would be


#### Example Output

``` 
0.8820970950608722  0.7904155783217401		0.6598575839917008 0.8822161621354134	 0.03114062354821738
```

Here, the first four annotators are fairly reliable, but the 5th one is not.


### 3. Entropies (`<prefix>.entropies`) - Optional

- One entropy value per line (if `--entropies` is used)
- Higher entropy = more uncertainty/disagreement among annotators, often more difficult items
- Lower entropy = high confidence/agreement

This will output a file with the same number of lines as the input file

#### Example Output

```
0.0027237895900081095
5.657170773284981E-5

0.009138546784668605
0.005036498835041038
0.0027237895900081095
```
Here, the first line after the break is the most difficult.


## Examples

### Basic Usage

```bash
# Evaluate annotations and write output to "prediction" and "competence"
python3 mace.py example.csv
```

### With Custom Prefix

```bash
# Write output to "out.prediction" and "out.competence"
python3 mace.py --prefix out example.csv
```

### Test Evaluation

```bash
# Evaluate against gold standard and print accuracy
python3 mace.py --test example.key example.csv
# Output: Accuracy on test set: 0.85
```

### Filter Uncertain Instances

```bash
# Only predict for top 90% most confident instances
python3 mace.py --threshold 0.9 example.csv
# Improves accuracy at the expense of coverage
```

### Continuous Numeric Values

```bash
# Process numeric scores, return weighted averages
python3 mace.py --continuous scores.csv

# With test evaluation (uses RMSE instead of accuracy)
python3 mace.py --continuous --test gold_standard.txt scores.csv
# Output: RMSE on test set: 2.345
```

### Distribution Output

```bash
# Get full probability distributions for each instance
python3 mace.py --distribution example.csv

# Discrete: "cat 0.8\tdog 0.15\tbird 0.05"
# Continuous: "3.5\t0.2\t2.0\t5.0\t3" (mean, std, min, max, n_annotators)
```

### With Control Items (Semi-Supervised)

```bash
# Use known labels to guide learning
python3 mace.py --controls known_labels.txt example.csv
```

### Output With Headers

```bash
# Add descriptive headers to output files
python3 mace.py --headers --prefix results example.csv
```

### Regular EM (Maximum Likelihood)

```bash
# Use regular EM instead of Variational Bayes EM (old default behavior)
python3 mace.py --em example.csv
```

### With Label Priors

```bash
# Use label priors file (priors.txt format: "label\tweight" one per line)
python3 mace.py --priors priors.txt example.csv

# Example priors.txt:
# cat	0.5
# dog	0.3
# bird	0.2
```

### Complete Example

```bash
# Full-featured run with all options
python3 mace.py \
    --continuous \
    --distribution \
    --headers \
    --entropies \
    --test gold_standard.txt \
    --controls known_labels.txt \
    --prefix results \
    --threshold 0.8 \
    --iterations 100 \
    --restarts 20 \
    scores.csv
```

## Understanding the Output

### Competence Scores

Competence scores range from 0 to 1:
- **0.9-1.0**: Highly reliable annotator
- **0.7-0.9**: Good annotator
- **0.5-0.7**: Moderate reliability
- **<0.5**: Unreliable annotator (may be spamming)

### Entropy

Entropy measures uncertainty in predictions:
- **Low entropy (<0.5)**: High confidence, annotators agree
- **Medium entropy (0.5-1.5)**: Moderate uncertainty
- **High entropy (>1.5)**: High uncertainty, annotators disagree

### Threshold Filtering

The `--threshold` option filters instances by entropy:
- `--threshold 1.0`: All instances (default)
- `--threshold 0.8`: Top 80% most certain instances

## Algorithm Details

### EM Algorithm

MACE uses **Variational Bayes Expectation-Maximization** (default) or regular **EM (Maximum Likelihood Estimation)** to iteratively:
1. **E-step**: Compute posterior probabilities over true labels and expected counts of annotator behaviors
2. **M-step**: Update competence estimates and spamming strategies
   - **Variational Bayes EM (default)**: Uses Bayesian priors with digamma function for normalization
   - **Regular EM**: Uses maximum likelihood with smoothing parameter

Use `--em` flag to switch to regular EM (the old default method).

### Model Parameters

- **spamming[a, 0]**: Probability that annotator `a` is guessing. Informed by alpha
- **spamming[a, 1]**: Probability that annotator `a` knows the answer. Informed by beta
- **thetas[a, l]**: Probability that annotator `a` guesses label `l` when spamming


### Continuous Mode

In continuous mode, MACE:
1. Still runs Variational Bayes EM to estimate annotator competence
2. Returns weighted averages: `Σ(value_i × competence_i) / Σ(competence_i)`
3. Uses RMSE for evaluation instead of accuracy

## Citation

If you use MACE in your research, please cite:
* *Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, and Eduard Hovy.* (2013). **Learning Whom to Trust With MACE.**  In: Proceedings of NAACL-HLT. Association for Computational Linguistics. [[PDF]](https://aclanthology.org/N13-1132.pdf)

```bib
@inproceedings{hovy-etal-2013-learning,
    title = "Learning Whom to Trust with {MACE}",
    author = "Hovy, Dirk  and
      Berg-Kirkpatrick, Taylor  and
      Vaswani, Ashish  and
      Hovy, Eduard",
    editor = "Vanderwende, Lucy  and
      Daum{\'e} III, Hal  and
      Kirchhoff, Katrin",
    booktitle = "Proceedings of the 2013 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2013",
    address = "Atlanta, Georgia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N13-1132/",
    pages = "1120--1130"
}
```

An additional paper compares MACE with some other annotation models:
* *Silviu Paun, Bob Carpenter, Jon Chamberlain, Dirk Hovy, Udo Kruschwitz, and Massimo Poesio.* (2018): **Comparing Bayesian Models of Annotation**. In: Transactions of the Association for Computational Linguistics (TACL). [[PDF]](https://aclanthology.org/Q18-1040.pdf)

```bib
@article{paun-etal-2018-comparing,
    title = "Comparing {B}ayesian Models of Annotation",
    author = "Paun, Silviu  and
      Carpenter, Bob  and
      Chamberlain, Jon  and
      Hovy, Dirk  and
      Kruschwitz, Udo  and
      Poesio, Massimo",
    editor = "Lee, Lillian  and
      Johnson, Mark  and
      Toutanova, Kristina  and
      Roark, Brian",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1040/",
    doi = "10.1162/tacl_a_00040",
    pages = "571--585",
    abstract = "The analysis of crowdsourced annotations in natural language processing is concerned with identifying (1) gold standard labels, (2) annotator accuracies and biases, and (3) item difficulties and error patterns. Traditionally, majority voting was used for 1, and coefficients of agreement for 2 and 3. Lately, model-based analysis of corpus annotations have proven better at all three tasks. But there has been relatively little work comparing them on the same datasets. This paper aims to fill this gap by analyzing six models of annotation, covering different approaches to annotator ability, item difficulty, and parameter pooling (tying) across annotators and items. We evaluate these models along four aspects: comparison to gold labels, predictive accuracy for new annotations, annotator characterization, and item difficulty, using four datasets with varying degrees of noise in the form of random (spammy) annotators. We conclude with guidelines for model selection, application, and implementation."
}

```

## Version

Current version: **0.3**

Python port of the Java implementation, modified to work with Python 3.12+.

## License

Copyright (c) 2013 by the University of Southern California. All rights reserved.

## Support

This is research software that is not actively maintained. For questions, please refer to the original paper or contact the authors.

## Troubleshooting

### Common Issues

1. **"No module named 'numpy'"**
   - Solution: Install dependencies with `pip install numpy scipy`

2. **"No annotators found in CSV file"**
   - Solution: Check that your CSV file has at least one column with data

3. **"Number of annotations in line X differs from previous line"**
   - Solution: Ensure all rows have the same number of columns (commas)

4. **"non-numeric value in test file (continuous mode requires numeric values)"**
   - Solution: In continuous mode, all values must be valid numbers

5. **File encoding issues**
   - Solution: Ensure your CSV file is saved in UTF-8 encoding

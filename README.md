# MACE (Multi-Annotator Competence Estimation)

When evaluating redundant annotations (like those from Amazon's MechanicalTurk), we usually want to 
1. aggregate annotations to recover the most likely answer
2. find out which annotators are trustworthy
3. evaluate item and task difficulty

MACE solves all of these problems, by learning competence estimates for each annotators and computing the most likely answer based on those competences.


## 1 Usage

(lines starting with '$' denote command line input)

### Shell script:

`$./MACE [options] <CSV input file>`
			
### Java

`$java -jar MACE.jar [options] <CSV input file>`

If you have trouble running the shell script, you might want to modify the script by adding your classpath and/or minimum and maximum heap space information.
	
MACE runs Variational Bayes EM training by default. If you would like vanilla EM training, set --em 


### Options:
```
	--controls <FILE>:	supply a file with annotated control items. Each line corresponds to one item,
				so the number of lines MUST match the input CSV file.
				The control items serve as semi-supervised input. Controls usually improve accuracy.

	--alpha <FLOAT>:	first hyper-parameter of beta prior that controls whether an annotator knows or guesses. Default:0.5, if --beta is set

	--beta <FLOAT>:		second hyper-parameter of beta prior that controls whether an annotator knows or guesses. Default:0.5, if --alpha is set

	--distribution:		for each items, list all labels and their probability in '[prefix.]prediction'

	--entropies:		write the entropy of each instance to a separate file '[prefix.]entropy'

	--help:			display this information

	--iterations <1-1000>:	number of iterations for each EM start. Default: 50

	--prefix <STRING>:	prefix used for output files.

	--priors <FILE>:	file with one label and weight pair per line (tab-separated). Must include all labels
		 		     in data file. Weights will be automatically normalized

	--restarts <1-1000>:	number of random restarts to perform. Default: 10

	--smoothing <0.0-1.0>:	smoothing added to fractional counts before normalization.
				Higher values mean smaller changes. Default: 0.01/|values|

	--test <FILE>:		supply a test file. Each line corresponds to one item in the CSV file,
				so the number of lines must match. If a test file is supplied,
				MACE outputs the accuracy of the predictions

	--threshold <0.0-1.0>:	only predict the label for instances whose entropy is among the top n%, ignore others.
				Thus '--threshold 0.0' will ignore all instances, '--threshold 1.0' includes all.
				This improves accuracy at the expense of coverage. Default: 1.0
```

## 2 Inputs

### Input File
The **input file** has to be a comma-separated file, where each line represents an item, and each column represents an annotator. Since version 0.3, MACE can also handle blank lines, as you might have when annotating sequential data (each word on one line, sentences separated by a blank line).

Missing annotations by an annotator on an item are represented by the empty string. Files should be formatted in UTF-8 to avoid problems with newline characters.
	
Examples:

1.: File with binary decisions:
```
0,1,,,,1,0,0
,,1,1,,0,0,1
1,0,0,1,,1,,0
```

2.: File with sequential POS annotations:
```
NOUN,,,NOUN,PRON
VERB,VERB,,VERB,

ADJ,,ADJ,,ADV
,VERB,,VERB,ADV
NOUN,,,NOUN,PRON
```	

Make sure the last line has a line break!


### Label Priors
The **prior** file is optional, and gives the a priori prevalence of the individual labels. We can supply this to MACE with `--priors priors.tsv` if we know the prior distribution, and it will take them into account. The file needs to list all labels (one per line) and tab-separated the weight, probability, or frequency (MACE automatically normalizes these).

Example:
```
NOUN	30
VERB	30
ADJ	20
ADV	10
PRON	10
```

### Control Items
If we know the correct answer for some items, we can include **control items**. This helps MACE assess annotator reliability. The file with control items needs to have the same number of lines as the input file, with the correct specified for the control items.

Example:
```
PRON






```


### Test File
If we know *all* answers and only want to get an accuray for MACE, we can supply a **test file** via `--test test.txt`. This file must have the same number of lines as the input file. MACE will output an accuracy score. This will not work when `--distribution` is set!

Example:
```
PRON
VERB

ADJ
VERB
NOUN
```


## 3 Outputs

MACE provides two standard output files:
* the most likely answer **prediction** for each item, `[prefix.]prediction`. This file has the same number of lines as the input file. Each line is the most likely answer value for the corresponding item. If you set --distribution, each line contains the distribution over answer values sorted by entropy. In the POS example from above, these files would look like this:
```
NOUN
VERB

ADJ
VERB
NOUN
```

or

```
NOUN 0.9997443833265887	PRON 7.140381903855615E-5	ADJ 6.140428479093134E-5	VERB 6.140428479093134E-5	ADV 6.140428479093134E-5
VERB 0.9999961943848287	NOUN 9.514037928812883E-7	ADJ 9.514037928812883E-7	PRON 9.514037928812883E-7	ADV 9.514037928812883E-7

ADJ 0.9990184050335877	ADV 2.741982824057974E-4	NOUN 2.3579889466878394E-4	VERB 2.3579889466878394E-4	PRON 2.3579889466878394E-4
VERB 0.9994950838119411	ADV 1.4104305366466138E-4	NOUN 1.2129104479807625E-4	ADJ 1.2129104479807625E-4	PRON 1.2129104479807625E-4
NOUN 0.9997443833265887	PRON 7.140381903855615E-5	ADJ 6.140428479093134E-5	VERB 6.140428479093134E-5	ADV 6.140428479093134E-5
```
* the **competence estimate** for each annotator, `[prefix.]competence`. This file has one line with tab separated values. In the POS example from above, this would be

```0.8820970950608722  0.7904155783217401		0.6598575839917008 0.8822161621354134	 0.03114062354821738```

* In addition, you can output the entropy of each item by setting `--entropies`. This will output a file with the same number of lines as the input file, named `[prefix.]entropy`. The output looks like this:
```
0.0027237895900081095
5.657170773284981E-5

0.009138546784668605
0.005036498835041038
0.0027237895900081095
```


## 4 Examples

`$java -jar MACE.jar example.csv`
Evaluate the file example.csv and write the output to "competence" and "prediction".

`$java -jar MACE.jar --prefix out example.csv`
Evaluate the file example.csv and write the output to "out.competence" and "out.prediction".

`$java -jar MACE.jar --prefix out --distribution example.csv`
Evaluate the file example.csv and write the output to "out.competence" and "out.prediction". For each item, show the distribution over answer values sorted by entropy.

`$java -jar MACE.jar --test example.key example.csv`
Evaluate the file example.csv against the true answers in example.key. 
Write the output to "competence" and "prediction" and print the accuracy to STDOUT (acc=0.8)

`$java -jar MACE.jar --threshold 0.9 example.csv`
Evaluate the file example.csv. Return predictions only for the 90% of items the model is most confident in (acc=0.84). 
Write the output to "competence" and "prediction". The latter will have blank lines for ignored items. 

`$java -jar MACE.jar --threshold 0.9 example.csv`
Evaluate the file example.csv. Return predictions only for the top 90% of items the model is most confident in. 
Write the output to "competence" and "prediction". The latter will have blank lines for ignored items. 
Compute the accuracy of only the predicted items and write to STDOUT.


## 5 References

To cite MACE in publications, please refer to:
* *Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, and Eduard Hovy* (2013): **Learning Whom to Trust With MACE**. In: Proceedings of NAACL-HLT. [PDF](http://www.aclweb.org/anthology/N13-1132)

```bib
@inproceedings{hovy2013learning,
  title={Learning whom to trust with MACE},
  author={Hovy, Dirk and Berg-Kirkpatrick, Taylor and Vaswani, Ashish and Hovy, Eduard},
  booktitle={Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={1120--1130},
  year={2013}
}

```

There is an additional paper that compares MACE with some other models:
* *Silviu Paun, Bob Carpenter, Jon Chamberlain, Dirk Hovy, Udo Kruschwitz, and Massimo Poesio* (2018): **Comparing Bayesian Models of Annotation**. In: Transactions of the Association for Computational Linguistics (TACL).
